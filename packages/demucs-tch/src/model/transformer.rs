//! Cross-domain transformer (tch-rs implementation).
//!
//! Ported from `demucs-core/src/model/transformer.rs` (Burn). Weights are kept
//! in PyTorch-native layout (packed `in_proj_weight` / `in_proj_bias`), so the
//! attention is implemented manually with `tch::Tensor` ops.

use tch::{Device, Kind, Tensor};

use crate::model::conv::LayerScale;
use crate::{T_HEADS, T_HIDDEN_SCALE, T_LAYERS};

fn gelu(x: &Tensor) -> Tensor {
    x.gelu("none")
}

/// A LayerNorm bundle (PyTorch `.weight` / `.bias`).
pub struct LayerNorm {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub d_model: i64,
}

impl LayerNorm {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.layer_norm(
            [self.d_model],
            self.weight.as_ref(),
            self.bias.as_ref(),
            1e-5,
            false,
        )
    }
}

/// A GroupNorm bundle (MyGroupNorm: 1 group over d_model channels).
pub struct GroupNorm {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl GroupNorm {
    /// x: [B, D, S] → group_norm → [B, D, S]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.group_norm(1, self.weight.as_ref(), self.bias.as_ref(), 1e-5, false)
    }
}

/// A Linear layer (PyTorch weight `[out, in]`).
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.linear(&self.weight, self.bias.as_ref())
    }
}

/// A 1x1 Conv1d used for channel up/down-sampling.
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Conv1d {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.conv1d(&self.weight, self.bias.as_ref(), [1], [0], [1], 1)
    }
}

/// PyTorch-packed MultiHeadAttention weights.
pub struct MultiHeadAttention {
    /// [3*d, d] packed Q, K, V
    pub in_proj_weight: Tensor,
    /// [3*d]
    pub in_proj_bias: Option<Tensor>,
    pub out_proj: Linear,
    pub n_heads: i64,
    pub d_model: i64,
}

impl MultiHeadAttention {
    /// PyTorch `nn.MultiheadAttention` semantics: packed `in_proj_weight` is
    /// `[q; k; v]` along dim 0, each `[d, d]`. Q projects `query`, K projects
    /// `key`, V projects `value` (each with its own slice). Batch-first, no dropout.
    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        let d = self.d_model;
        let h = self.n_heads;
        let head_dim = d / h;
        let scaling = 1.0 / (head_dim as f64).sqrt();

        // Split packed projection into Q/K/V slices (PyTorch layout [d, d]).
        let w_q = self.in_proj_weight.narrow(0, 0, d);
        let w_k = self.in_proj_weight.narrow(0, d, d);
        let w_v = self.in_proj_weight.narrow(0, 2 * d, d);
        let (b_q, b_k, b_v) = match &self.in_proj_bias {
            Some(b) => (
                Some(b.narrow(0, 0, d)),
                Some(b.narrow(0, d, d)),
                Some(b.narrow(0, 2 * d, d)),
            ),
            None => (None, None, None),
        };

        let q = query.linear(&w_q, b_q.as_ref());
        let k = key.linear(&w_k, b_k.as_ref());
        let v = value.linear(&w_v, b_v.as_ref());

        let b = q.size()[0];
        let sq = q.size()[1];
        let sk = k.size()[1];

        // [B, S, D] -> [B, H, S, head_dim]
        let q = q.reshape([b, sq, h, head_dim]).permute([0, 2, 1, 3]);
        let k = k.reshape([b, sk, h, head_dim]).permute([0, 2, 1, 3]);
        let v = v.reshape([b, sk, h, head_dim]).permute([0, 2, 1, 3]);

        // q · k^T * scaling → softmax → · v
        let attn = q.matmul(&k.transpose(-2, -1)) * scaling;
        let attn = attn.softmax(-1, Kind::Float);
        let ctx = attn.matmul(&v); // [B, H, Sq, head_dim]

        // [B, H, Sq, head_dim] -> [B, Sq, D]
        let ctx = ctx.permute([0, 2, 1, 3]).reshape([b, sq, d]);
        self.out_proj.forward(&ctx)
    }

    fn self_attn(&self, x: &Tensor) -> Tensor {
        self.forward(x, x, x)
    }

    fn cross_attn(&self, query: &Tensor, cross: &Tensor) -> Tensor {
        self.forward(query, cross, cross)
    }
}

pub struct SelfAttentionLayer {
    pub norm1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub gamma_1: LayerScale,
    pub norm2: LayerNorm,
    pub linear1: Linear,
    pub linear2: Linear,
    pub gamma_2: LayerScale,
    pub norm_out: Option<GroupNorm>,
}

impl SelfAttentionLayer {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let residual = x.shallow_clone();
        let x = self.norm1.forward(x);
        let attn_out = self.attn.self_attn(&x);
        let x = self.gamma_1.forward_last(&attn_out) + residual;

        let residual = x.shallow_clone();
        let x = self.norm2.forward(&x);
        let x = gelu(&self.linear1.forward(&x));
        let x = self.linear2.forward(&x);
        let x = self.gamma_2.forward_last(&x) + residual;

        match &self.norm_out {
            Some(norm) => {
                let x = x.transpose(1, 2);
                let x = norm.forward(&x);
                x.transpose(1, 2)
            }
            None => x,
        }
    }
}

pub struct CrossAttentionLayer {
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub attn: MultiHeadAttention,
    pub gamma_1: LayerScale,
    pub norm3: LayerNorm,
    pub linear1: Linear,
    pub linear2: Linear,
    pub gamma_2: LayerScale,
    pub norm_out: Option<GroupNorm>,
}

impl CrossAttentionLayer {
    pub fn forward(&self, query: &Tensor, cross: &Tensor) -> Tensor {
        let residual = query.shallow_clone();
        let q = self.norm1.forward(query);
        let kv = self.norm2.forward(cross);
        let attn_out = self.attn.cross_attn(&q, &kv);
        let x = self.gamma_1.forward_last(&attn_out) + residual;

        let residual = x.shallow_clone();
        let x = gelu(&self.linear1.forward(&self.norm3.forward(&x)));
        let x = self.linear2.forward(&x);
        let x = self.gamma_2.forward_last(&x) + residual;

        match &self.norm_out {
            Some(norm) => {
                let x = x.transpose(1, 2);
                let x = norm.forward(&x);
                x.transpose(1, 2)
            }
            None => x,
        }
    }
}

pub enum TransformerLayer {
    SelfAttn(SelfAttentionLayer),
    CrossAttn(CrossAttentionLayer),
}

pub struct CrossDomainTransformer {
    pub d_model: i64,
    pub norm_in: LayerNorm,
    pub norm_in_t: LayerNorm,
    pub channel_upsampler: Option<Conv1d>,
    pub channel_downsampler: Option<Conv1d>,
    pub channel_upsampler_t: Option<Conv1d>,
    pub channel_downsampler_t: Option<Conv1d>,
    pub layers: Vec<TransformerLayer>,
    pub layers_t: Vec<TransformerLayer>,
}

impl CrossDomainTransformer {
    /// freq: [1, ch, Fr, T], time: [1, ch, Tt]
    pub fn forward(&self, freq: &Tensor, time: &Tensor) -> (Tensor, Tensor) {
        let fdim = freq.size();
        let (ch, freq_bins, time_f) = (fdim[1], fdim[2], fdim[3]);
        let time_t = time.size()[2];
        let d = self.d_model;

        // 1. Channel upsample
        let freq = match &self.channel_upsampler {
            Some(up) => {
                let t = freq
                    .reshape([ch, freq_bins, time_f])
                    .transpose(0, 1); // [Fr, ch, T]
                let t = up.forward(&t); // [Fr, d, T]
                t.transpose(0, 1).unsqueeze(0) // [1, d, Fr, T]
            }
            None => freq.shallow_clone(),
        };
        let time = match &self.channel_upsampler_t {
            Some(up) => up.forward(time),
            None => time.shallow_clone(),
        };

        // 2. Flatten freq time-major, reshape time
        let freq = freq
            .permute([0, 3, 2, 1])
            .reshape([1, time_f * freq_bins, d]);
        let time = time.permute([0, 2, 1]);

        // 3. Norm FIRST, then add positional embeddings (Python order)
        let freq = self.norm_in.forward(&freq);
        let time = self.norm_in_t.forward(&time);
        let device = freq.device();
        let freq =
            freq + create_2d_sin_embed(d as usize, freq_bins as usize, time_f as usize).to_device(device);
        let mut time = time + create_sin_embed(time_t as usize, d as usize).to_device(device);

        // 4. Transformer layers
        let mut freq = freq;
        for (layer_f, layer_t) in self.layers.iter().zip(self.layers_t.iter()) {
            match (layer_f, layer_t) {
                (TransformerLayer::SelfAttn(f), TransformerLayer::SelfAttn(t)) => {
                    freq = f.forward(&freq);
                    time = t.forward(&time);
                }
                (TransformerLayer::CrossAttn(f), TransformerLayer::CrossAttn(t)) => {
                    let freq_old = freq.shallow_clone();
                    let new_freq = f.forward(&freq_old, &time.shallow_clone());
                    let new_time = t.forward(&time, &freq_old);
                    freq = new_freq;
                    time = new_time;
                }
                _ => panic!("freq/time transformer layer types must match"),
            }
        }

        // 5. Unflatten
        let freq = freq
            .reshape([1, time_f, freq_bins, d])
            .permute([0, 3, 2, 1]);
        let time = time.permute([0, 2, 1]);

        // 6. Channel downsample
        let freq = match &self.channel_downsampler {
            Some(down) => {
                let t = freq.reshape([d, freq_bins, time_f]).transpose(0, 1);
                let t = down.forward(&t);
                t.transpose(0, 1).unsqueeze(0)
            }
            None => freq,
        };
        let time = match &self.channel_downsampler_t {
            Some(down) => down.forward(&time),
            None => time,
        };

        (freq, time)
    }
}

/// 1D sinusoidal embedding `[1, seq_len, d_model]`, cos-first half then sin.
pub fn create_sin_embed(seq_len: usize, d_model: usize) -> Tensor {
    let mut data = vec![0.0f32; seq_len * d_model];
    let half = d_model / 2;
    let half_m1 = if half > 1 { (half - 1) as f32 } else { 1.0 };
    for pos in 0..seq_len {
        for i in 0..half {
            let angle = pos as f32 / (10000.0_f32).powf(i as f32 / half_m1);
            data[pos * d_model + i] = angle.cos();
            data[pos * d_model + half + i] = angle.sin();
        }
    }
    Tensor::from_slice(&data)
        .reshape([seq_len as i64, d_model as i64])
        .unsqueeze(0)
        .to_kind(Kind::Float)
}

/// 2D sinusoidal embedding `[1, width*height, d_model]`, time-major.
pub fn create_2d_sin_embed(d_model: usize, height: usize, width: usize) -> Tensor {
    let half = d_model / 2;
    let quarter = half / 2;
    let div_terms: Vec<f32> = (0..quarter)
        .map(|k| (-2.0 * k as f32 * (10000.0_f32).ln() / half as f32).exp())
        .collect();

    let seq_len = width * height;
    let mut data = vec![0.0f32; seq_len * d_model];
    for t in 0..width {
        for fr in 0..height {
            let s = t * height + fr;
            for k in 0..quarter {
                let w_angle = t as f32 * div_terms[k];
                let h_angle = fr as f32 * div_terms[k];
                data[s * d_model + 2 * k] = w_angle.sin();
                data[s * d_model + 2 * k + 1] = w_angle.cos();
                data[s * d_model + half + 2 * k] = h_angle.sin();
                data[s * d_model + half + 2 * k + 1] = h_angle.cos();
            }
        }
    }
    Tensor::from_slice(&data)
        .reshape([seq_len as i64, d_model as i64])
        .unsqueeze(0)
        .to_kind(Kind::Float)
}

/// Build the (self, cross, self, cross, self) layer pattern metadata.
pub fn layer_is_self(i: usize) -> bool {
    i % 2 == 0
}

pub fn ffn_dim(d_model: usize) -> usize {
    (d_model as f32 * T_HIDDEN_SCALE as f32) as usize
}

pub const N_LAYERS: usize = T_LAYERS;
pub const N_HEADS: i64 = T_HEADS as i64;

pub fn device_of(t: &Tensor) -> Device {
    t.device()
}

/// Wrap a raw scale tensor into a conv::LayerScale (shared type).
pub fn create_layer_scale(scale: Tensor) -> LayerScale {
    LayerScale { scale }
}
