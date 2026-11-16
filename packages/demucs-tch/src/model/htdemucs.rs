//! HTDemucs model (tch-rs implementation).
//!
//! Ported from `demucs-core/src/model/htdemucs.rs` (Burn).

use tch::{Kind, Tensor};

use crate::model::conv::{HDecLayer, HEncLayer, TDecLayer, TEncLayer};
use crate::model::metadata::ModelInfo;
use crate::model::transformer::CrossDomainTransformer;
use crate::{CHANNELS, DEPTH, GROWTH, N_FFT};

pub struct HyperParameters {
    pub n_sources: usize,
    pub bottom_channels: usize,
}

impl HyperParameters {
    pub fn from_model_info(info: &ModelInfo) -> Self {
        let n_sources = info.stems.len();
        let bottom_channels = if n_sources == 6 {
            CHANNELS * GROWTH.pow(DEPTH as u32 - 1)
        } else {
            512
        };
        Self {
            n_sources,
            bottom_channels,
        }
    }
}

pub struct HTDemucs {
    pub encoders: Vec<HEncLayer>,
    pub tencoders: Vec<TEncLayer>,
    pub crosstransformer: CrossDomainTransformer,
    pub decoders: Vec<HDecLayer>,
    pub tdecoders: Vec<TDecLayer>,
    /// [N_FFT/2, first_chout] embedding weights (already ×10 baked in at load).
    pub freq_emb_weight: Tensor,
    pub n_sources: usize,
}

impl HTDemucs {
    /// freq: [1, 4, 2048, T] (CaC), time: [1, 2, samples].
    /// Returns (freq_out [1, n_sources*4, 2048, T], time_out [1, n_sources*2, samples]).
    pub fn forward(&self, freq: &Tensor, time: &Tensor) -> (Tensor, Tensor) {
        // 1. Normalize inputs (biased variance, matching Burn var_mean)
        let (freq, freq_mean, freq_std) = Self::normalize_freq(freq);
        let (time, time_mean, time_std) = Self::normalize_time(time);

        // 2. Encode freq
        let mut freq_skips: Vec<Tensor> = Vec::new();
        let mut f = self.encoders[0].forward(&freq);

        // Apply freq_emb AFTER encoder[0], BEFORE saving skip.
        let fr = f.size()[2];
        let frs = Tensor::arange(fr, (Kind::Int64, f.device())).unsqueeze(0); // [1, Fr]
        let emb = Tensor::embedding(
            &self.freq_emb_weight,
            &frs,
            -1,
            false,
            false,
        ); // [1, Fr, chout]
        let emb = emb.permute([0, 2, 1]).unsqueeze(3); // [1, chout, Fr, 1]
        f = f + emb * 0.2;
        freq_skips.push(f.shallow_clone());

        for enc in &self.encoders[1..] {
            f = enc.forward(&f);
            freq_skips.push(f.shallow_clone());
        }

        // 3. Encode time (save input lengths for decoder trimming)
        let mut time_skips: Vec<Tensor> = Vec::new();
        let mut time_lengths: Vec<i64> = Vec::new();
        let mut t = time;
        for enc in &self.tencoders {
            time_lengths.push(t.size()[2]);
            t = enc.forward(&t);
            time_skips.push(t.shallow_clone());
        }

        // 4. Bottleneck
        let (mut f, mut t) = self.crosstransformer.forward(&f, &t);

        // 5. Decode freq (reverse skips)
        let freq_dims: Vec<i64> = freq_skips.iter().map(|s| s.size()[2]).collect();
        for (i, dec) in self.decoders.iter().enumerate() {
            let skip = freq_skips.pop().expect("freq skip stack exhausted");
            let freq_target = if i + 1 < freq_dims.len() {
                freq_dims[freq_dims.len() - 2 - i]
            } else {
                (N_FFT / 2) as i64
            };
            f = dec.forward(&f, &skip, freq_target);
        }

        // 6. Decode time (reverse skips)
        for (i, dec) in self.tdecoders.iter().enumerate() {
            let skip = time_skips.pop().expect("time skip stack exhausted");
            let time_target = time_lengths[time_lengths.len() - 1 - i];
            t = dec.forward(&t, &skip, time_target);
        }

        // 7. Denormalize (raw std, NO eps)
        let f = f * &freq_std + &freq_mean;
        let t = t * &time_std + &time_mean;
        (f, t)
    }

    /// Returns (normalized, mean [B,1,1,1], std [B,1,1,1]).
    fn normalize_freq(freq: &Tensor) -> (Tensor, Tensor, Tensor) {
        let d = freq.size();
        let b = d[0];
        let flat = freq.reshape([b, d[1] * d[2] * d[3]]);
        let mean = flat.mean_dim(1, true, Kind::Float); // [B, 1]
        let var = (&flat - &mean).pow_tensor_scalar(2.0).mean_dim(1, true, Kind::Float);
        let std = var.sqrt();
        let mean4 = mean.unsqueeze(2).unsqueeze(3);
        let std4 = std.unsqueeze(2).unsqueeze(3);
        let normalized = (freq - &mean4) / (&std4 + 1e-5);
        (normalized, mean4, std4)
    }

    /// Returns (normalized, mean [B,1,1], std [B,1,1]).
    fn normalize_time(time: &Tensor) -> (Tensor, Tensor, Tensor) {
        let d = time.size();
        let b = d[0];
        let flat = time.reshape([b, d[1] * d[2]]);
        let mean = flat.mean_dim(1, true, Kind::Float); // [B, 1]
        let var = (&flat - &mean).pow_tensor_scalar(2.0).mean_dim(1, true, Kind::Float);
        let std = var.sqrt();
        let mean3 = mean.unsqueeze(2);
        let std3 = std.unsqueeze(2);
        let normalized = (time - &mean3) / (&std3 + 1e-5);
        (normalized, mean3, std3)
    }
}
