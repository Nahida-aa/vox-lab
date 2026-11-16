//! demucs-tch — Burn-free Demucs inference via tch-rs (libtorch CPU).
//!
//! Ported from `submodule/demucs-rs/demucs-core` (Burn implementation). The pure-Rust
//! DSP / metadata / tensor-store code is copied verbatim (see module docs); all learned
//! modules and the pipeline are reimplemented directly on `tch::Tensor`.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use tch::{Device, Kind, Tensor};

use dsp::cac::{cac_data_to_complex, stft_to_cac};
use dsp::resample::resample_channel;
use dsp::stft::Stft;
use model::metadata::{ModelInfo, StemId};
use model::htdemucs::HTDemucs;
use provider::{fs::FsProvider, ModelProvider};
use weights::load::load_model;

pub mod dsp;
pub mod error;
pub mod listener;
pub mod model;
pub mod provider;
pub mod weights;

// ---------------------------------------------------------------------------
// Constants (from demucs-core/src/lib.rs)
// ---------------------------------------------------------------------------

pub const AUDIO_CHANNELS: usize = 2;
pub const N_FFT: usize = 4096;
pub const HOP_LENGTH: usize = 1024;
pub const CHANNELS: usize = 48;
pub const GROWTH: usize = 2;
pub const DEPTH: usize = 4;
pub const KERNEL_SIZE: usize = 8;
pub const STRIDE: usize = 4;
pub const T_LAYERS: usize = 5;
pub const T_HEADS: usize = 8;
pub const T_HIDDEN_SCALE: f32 = 4.0;
pub const DCONV_COMP: usize = 8;
pub const DCONV_DEPTH: usize = 2;
pub const SAMPLE_RATE: usize = 44100;
pub const TRAINING_LENGTH: usize = 343980;

pub const DEFAULT_MODEL_DIR: &str = "data/models/demucs";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "demucs-tch")]
pub struct Cli {
    /// Only load the model and print load time, then exit.
    #[arg(long)]
    pub benchmark_load: bool,

    /// Input WAV file.
    pub input: Option<PathBuf>,

    /// Output directory for stems.
    pub output: Option<PathBuf>,

    /// Model variant.
    #[arg(short, long, default_value = "htdemucs_ft")]
    pub model: String,

    /// Model directory (relative to repo root, or absolute).
    #[arg(long, env = "DEMUCS_MODEL_DIR", default_value = DEFAULT_MODEL_DIR)]
    pub model_dir: String,

    /// No-op for the tch backend (kept for CLI compatibility).
    #[arg(long, default_value = "128")]
    pub tasks_max: u32,

    /// No-op for the tch backend (kept for CLI compatibility).
    #[arg(long)]
    pub warmup: bool,

    /// Number of in-process inference rounds.
    #[arg(long, default_value = "1")]
    pub benchmark_rounds: u32,
}

// ---------------------------------------------------------------------------
// Path resolution (mirrors demucs-burn/src/lib.rs:57-77)
// ---------------------------------------------------------------------------

/// Walk up from the executable to the repo root. The exact depth differs by
/// layout (`vox-lab/target/release/<bin>` vs the standalone-workspace layout
/// `vox-lab/packages/demucs-tch/target/<profile>/<bin>`), so we look for the
/// directory that owns both `Cargo.toml` and a `packages/` subdir. Does not
/// depend on the current working directory.
pub fn current_exe_repo_root() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("current_exe failed")?;
    let mut dir = exe
        .parent()
        .context("could not resolve exe parent")?
        .to_path_buf();
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("packages").is_dir() {
            return Ok(dir);
        }
        dir = match dir.parent() {
            Some(p) => p.to_path_buf(),
            None => bail!("could not locate repo root from {}", exe.display()),
        };
    }
}

pub fn resolve_path(root: &Path, p: &str) -> PathBuf {
    let path = Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

pub fn resolve_model_info(name: &str) -> Result<&'static ModelInfo> {
    model::metadata::ALL_MODELS
        .iter()
        .copied()
        .find(|m| m.id == name)
        .with_context(|| format!("unknown model '{}'", name))
}

// ---------------------------------------------------------------------------
// Model container + inference pipeline
// ---------------------------------------------------------------------------

pub struct Demucs {
    pub info: &'static ModelInfo,
    pub models: Vec<HTDemucs>,
    pub device: Device,
}

impl Demucs {
    pub fn from_bytes(info: &'static ModelInfo, bytes: &[u8]) -> Result<Self> {
        // The linked libtorch decides what is available: CPU-only builds always
        // resolve to Cpu, CUDA/ROCm builds (torch-sys links torch_cuda/torch_hip
        // based on the libtorch that was found at build time) get Cuda(0).
        let device = Device::cuda_if_available();
        let models = load_model(bytes, info, device)
            .map_err(|e| anyhow::anyhow!("weight load failed: {}", e))?;
        Ok(Self {
            info,
            models,
            device,
        })
    }

    /// Full pipeline: stereo f32 input (at `sample_rate`) → per-stem stereo f32.
    pub fn separate(&self, left: &[f32], right: &[f32], sample_rate: u32) -> Result<Vec<Stem>> {
        let info = self.info;
        let needs_resample = sample_rate != SAMPLE_RATE as u32;
        let (left_in, right_in): (Cow<[f32]>, Cow<[f32]>) = if needs_resample {
            let l = resample_channel(left, sample_rate, SAMPLE_RATE as u32)
                .map_err(|e| anyhow::anyhow!("resample L: {}", e))?;
            let r = resample_channel(right, sample_rate, SAMPLE_RATE as u32)
                .map_err(|e| anyhow::anyhow!("resample R: {}", e))?;
            (Cow::Owned(l), Cow::Owned(r))
        } else {
            (Cow::Borrowed(left), Cow::Borrowed(right))
        };
        let left = &*left_in;
        let right = &*right_in;
        let n_samples = left.len();

        let mut stems = if n_samples <= TRAINING_LENGTH {
            self.separate_single_segment(left, right, n_samples)?
        } else {
            // Chunked inference with 25% overlap and triangular window.
            let segment = TRAINING_LENGTH;
            let stride = segment * 3 / 4;
            let num_chunks = n_samples.saturating_sub(segment).div_ceil(stride) + 1;
            let n_stems = info.stems.len();
            let mut out_left = vec![vec![0.0f32; n_samples]; n_stems];
            let mut out_right = vec![vec![0.0f32; n_samples]; n_stems];
            let mut sum_weight = vec![0.0f32; n_samples];

            for chunk_idx in 0..num_chunks {
                let pct = (chunk_idx * 100 / num_chunks).min(99);
                println!("({}%)", pct);

                let start = chunk_idx * stride;
                let end = (start + segment).min(n_samples);
                let chunk_len = end - start;
                let chunk_stems =
                    self.separate_single_segment(&left[start..end], &right[start..end], chunk_len)?;

                let window = triangular_window(chunk_len);
                for stem in &chunk_stems {
                    let s = info.stems.iter().position(|&id| id == stem.id).unwrap();
                    for i in 0..chunk_len {
                        let w = window[i];
                        out_left[s][start + i] += w * stem.left[i];
                        out_right[s][start + i] += w * stem.right[i];
                    }
                }
                for i in 0..chunk_len {
                    sum_weight[start + i] += window[i];
                }
                let pct = ((chunk_idx + 1) * 100 / num_chunks).min(100);
                println!("({}%)", pct);
            }

            let mut stems = Vec::with_capacity(n_stems);
            for (s, &stem_id) in info.stems.iter().enumerate() {
                for i in 0..n_samples {
                    let w = sum_weight[i];
                    if w > 0.0 {
                        out_left[s][i] /= w;
                        out_right[s][i] /= w;
                    }
                }
                stems.push(Stem {
                    id: stem_id,
                    left: std::mem::take(&mut out_left[s]),
                    right: std::mem::take(&mut out_right[s]),
                });
            }
            stems
        };

        if needs_resample {
            for stem in &mut stems {
                stem.left = resample_channel(&stem.left, SAMPLE_RATE as u32, sample_rate)
                    .map_err(|e| anyhow::anyhow!("resample back L: {}", e))?;
                stem.right = resample_channel(&stem.right, SAMPLE_RATE as u32, sample_rate)
                    .map_err(|e| anyhow::anyhow!("resample back R: {}", e))?;
            }
        }
        Ok(stems)
    }

    fn separate_single_segment(
        &self,
        left: &[f32],
        right: &[f32],
        n_samples: usize,
    ) -> Result<Vec<Stem>> {
        let info = self.info;
        let padded_len = TRAINING_LENGTH;
        let mut left_padded = vec![0.0f32; padded_len];
        let mut right_padded = vec![0.0f32; padded_len];
        left_padded[..n_samples].copy_from_slice(left);
        right_padded[..n_samples].copy_from_slice(right);

        let mut stft = Stft::new(N_FFT, HOP_LENGTH);
        let left_spec = stft
            .forward(&left_padded)
            .map_err(|e| anyhow::anyhow!("STFT L: {}", e))?;
        let right_spec = stft
            .forward(&right_padded)
            .map_err(|e| anyhow::anyhow!("STFT R: {}", e))?;
        let bins = N_FFT / 2;
        let n_frames = left_spec.len() / bins;

        let left_cac = stft_to_cac(&left_spec, N_FFT, self.device);
        let right_cac = stft_to_cac(&right_spec, N_FFT, self.device);
        let freq = Tensor::cat(&[left_cac, right_cac], 0).unsqueeze(0); // [1,4,F,T]
        let time = build_time_tensor(&left_padded, &right_padded, padded_len, self.device);

        let n_sources = info.stems.len();
        let mut stems = Vec::with_capacity(n_sources);

        if self.models.len() == 1 {
            // Single model (htdemucs / htdemucs_6s): all stems in one forward.
            let (freq_out, time_out) = self.models[0].forward(&freq, &time);
            stems.extend(extract_all_stems(
                &freq_out, &time_out, info, n_frames, padded_len, n_samples, &mut stft,
            )?);
        } else {
            // FineTuned: one sub-model per stem.
            for (i, &stem_id) in info.stems.iter().enumerate() {
                let (freq_out, time_out) = self.models[i].forward(&freq, &time);
                stems.push(extract_single_stem(
                    &freq_out, &time_out, i, stem_id, n_frames, padded_len, n_samples, &mut stft,
                )?);
            }
        }
        Ok(stems)
    }
}

pub struct Stem {
    pub id: StemId,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

fn triangular_window(length: usize) -> Vec<f32> {
    if length <= 1 {
        return vec![1.0; length];
    }
    let denom = (length - 1) as f32;
    (0..length)
        .map(|i| 1.0 - (2.0 * i as f32 / denom - 1.0).abs())
        .collect()
}

fn build_time_tensor(left: &[f32], right: &[f32], padded_len: usize, device: Device) -> Tensor {
    let mut data = vec![0.0f32; 2 * padded_len];
    data[..left.len()].copy_from_slice(left);
    data[padded_len..padded_len + right.len()].copy_from_slice(right);
    Tensor::from_slice(&data)
        .reshape([1, 2, padded_len as i64])
        .to_kind(Kind::Float)
        .to(device)
}

fn tensor_to_vec(t: &Tensor) -> Vec<f32> {
    let n: i64 = t.size().iter().product();
    let flat = t.reshape([n]).to_device(Device::Cpu);
    Vec::<f32>::try_from(&flat).unwrap_or_else(|_| {
        // Fallback: copy through f64 (tch 0.27 implements From<&Tensor> for Vec<f64>).
        Vec::<f64>::try_from(&flat)
            .unwrap()
            .iter()
            .map(|&v| v as f32)
            .collect()
    })
}

fn extract_all_stems(
    freq_out: &Tensor,
    time_out: &Tensor,
    info: &ModelInfo,
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Result<Vec<Stem>> {
    let n_sources = info.stems.len();
    let freq_bins = N_FFT / 2;

    let freq_trimmed = freq_out
        .squeeze_dim(0)
        .narrow(2, 0, n_frames as i64);
    let time_trimmed = time_out
        .squeeze_dim(0)
        .narrow(1, 0, n_samples as i64);

    let freq_data = tensor_to_vec(&freq_trimmed.reshape([(n_sources * 4 * freq_bins * n_frames) as i64]));
    let time_data = tensor_to_vec(&time_trimmed.reshape([(n_sources * 2 * n_samples) as i64]));

    let cac_stride = 4 * freq_bins * n_frames;
    let ch_stride = 2 * freq_bins * n_frames;
    let time_stride = 2 * n_samples;

    let mut stems = Vec::with_capacity(n_sources);
    for (i, &stem_id) in info.stems.iter().enumerate() {
        let off = i * cac_stride;
        let left_spec = cac_data_to_complex(&freq_data[off..off + ch_stride], freq_bins, n_frames);
        let right_spec =
            cac_data_to_complex(&freq_data[off + ch_stride..off + cac_stride], freq_bins, n_frames);
        let lf = stft
            .inverse(&left_spec, padded_len)
            .map_err(|e| anyhow::anyhow!("iSTFT L: {}", e))?;
        let rf = stft
            .inverse(&right_spec, padded_len)
            .map_err(|e| anyhow::anyhow!("iSTFT R: {}", e))?;

        let toff = i * time_stride;
        let lt = &time_data[toff..toff + n_samples];
        let rt = &time_data[toff + n_samples..toff + time_stride];

        let left: Vec<f32> = lf[..n_samples].iter().zip(lt).map(|(f, t)| f + t).collect();
        let right: Vec<f32> = rf[..n_samples].iter().zip(rt).map(|(f, t)| f + t).collect();
        stems.push(Stem { id: stem_id, left, right });
    }
    Ok(stems)
}

fn extract_single_stem(
    freq_out: &Tensor,
    time_out: &Tensor,
    stem_idx: usize,
    stem_id: StemId,
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Result<Stem> {
    let freq_bins = N_FFT / 2;
    let freq_stem = freq_out
        .narrow(1, (stem_idx * 4) as i64, 4)
        .narrow(3, 0, n_frames as i64)
        .squeeze_dim(0);
    let time_stem = time_out
        .narrow(1, (stem_idx * 2) as i64, 2)
        .narrow(2, 0, n_samples as i64)
        .squeeze_dim(0);

    let freq_data = tensor_to_vec(&freq_stem.reshape([(4 * freq_bins * n_frames) as i64]));
    let time_data = tensor_to_vec(&time_stem.reshape([(2 * n_samples) as i64]));

    let ch_stride = 2 * freq_bins * n_frames;
    let left_spec = cac_data_to_complex(&freq_data[..ch_stride], freq_bins, n_frames);
    let right_spec = cac_data_to_complex(&freq_data[ch_stride..], freq_bins, n_frames);
    let lf = stft
        .inverse(&left_spec, padded_len)
        .map_err(|e| anyhow::anyhow!("iSTFT L: {}", e))?;
    let rf = stft
        .inverse(&right_spec, padded_len)
        .map_err(|e| anyhow::anyhow!("iSTFT R: {}", e))?;
    let (lt, rt) = time_data.split_at(n_samples);

    let left: Vec<f32> = lf[..n_samples].iter().zip(lt).map(|(f, t)| f + t).collect();
    let right: Vec<f32> = rf[..n_samples].iter().zip(rt).map(|(f, t)| f + t).collect();
    Ok(Stem { id: stem_id, left, right })
}

// ---------------------------------------------------------------------------
// WAV I/O (from demucs-burn/src/lib.rs:206-249)
// ---------------------------------------------------------------------------

pub fn read_wav(path: &Path) -> Result<(Vec<f32>, Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV: {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels > 2 {
        bail!("Expected mono or stereo, got {} channels", spec.channels);
    }
    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => {
            let max = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max)
                .collect()
        }
    };
    if spec.channels == 1 {
        Ok((samples.clone(), samples, sample_rate))
    } else {
        let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
        let right: Vec<f32> = samples.iter().skip(1).step_by(2).copied().collect();
        Ok((left, right, sample_rate))
    }
}

pub fn write_wav(path: &Path, left: &[f32], right: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for (&l, &r) in left.iter().zip(right.iter()) {
        writer.write_sample(l)?;
        writer.write_sample(r)?;
    }
    writer.finalize()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Main entry. stdout contract (consumers depend on it, DO NOT change):
/// `(NN%)` progress, `Benchmark-Load-Time: {:.3}`, `Benchmark-Warmup-Time: {:.3}`,
/// `Benchmark-Gen-Time-Round{N}: {:.3}` and a bare `Benchmark-Gen-Time: {:.3}`.
/// All human-readable diagnostics go to stderr.
pub fn run(cli: Cli) -> Result<()> {
    if cli.benchmark_rounds == 0 {
        bail!("--benchmark-rounds must be >= 1");
    }
    let info = resolve_model_info(&cli.model)?;

    let root = current_exe_repo_root()?;
    let model_dir = resolve_path(&root, &cli.model_dir);
    let provider = FsProvider::with_dir(model_dir.clone());
    eprintln!("Loading cached model: {}", provider_path(info, &model_dir));

    let t_load = std::time::Instant::now();
    let bytes = provider
        .load_cached(info)
        .map_err(|e| anyhow::anyhow!("model not cached ({}): {}", provider_path(info, &model_dir), e))?;
    let demucs = Demucs::from_bytes(info, &bytes)?;
    let load_secs = t_load.elapsed().as_secs_f64();
    eprintln!("Device: {:?}", demucs.device);
    println!("Benchmark-Load-Time: {:.3}", load_secs);

    if cli.warmup {
        eprintln!("Skipping GPU warmup (tch/libtorch backend has no shader precompile)");
        println!("Benchmark-Warmup-Time: 0.000");
    }

    if cli.benchmark_load {
        return Ok(());
    }

    let input = cli.input.as_ref().context("Input file required")?;
    let output = cli.output.as_ref().context("Output directory required")?;
    std::fs::create_dir_all(output)?;

    let (left, right, sample_rate) = read_wav(input)?;
    eprintln!("Reading {} ({} Hz)", input.display(), sample_rate);

    let mut last_stems: Option<Vec<Stem>> = None;
    for round in 1..=cli.benchmark_rounds {
        eprintln!("--- Round {}/{} ---", round, cli.benchmark_rounds);
        let t0 = std::time::Instant::now();
        let stems = demucs.separate(&left, &right, sample_rate)?;
        let gen_secs = t0.elapsed().as_secs_f64();
        println!("Benchmark-Gen-Time-Round{}: {:.3}", round, gen_secs);
        if round == cli.benchmark_rounds {
            println!("Benchmark-Gen-Time: {:.3}", gen_secs);
            last_stems = Some(stems);
        }
    }
    println!("(100%)");

    let stems = last_stems.context("no inference round produced stems")?;
    for (i, stem) in stems.iter().enumerate() {
        let path = output.join(format!("target_{}_{}.wav", i, stem.id.as_str()));
        write_wav(&path, &stem.left, &stem.right, sample_rate)?;
        eprintln!("Wrote {}", path.display());
    }
    eprintln!("Done!");
    Ok(())
}

fn provider_path(info: &ModelInfo, model_dir: &Path) -> String {
    model_dir.join(info.filename).display().to_string()
}
