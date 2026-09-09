//! demucs-burn 公共逻辑。各后端二进制（`src/bin/demucs-burn-*`）是薄壳：
//! 初始化自己的 device 后调用 [`run`]。

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use burn::prelude::Backend;
use clap::Parser;

use demucs_core::listener::{ForwardEvent, ForwardListener};
use demucs_core::model::metadata::{
    ALL_MODELS, HTDEMUCS_6S_ID, HTDEMUCS_FT_ID, HTDEMUCS_ID, ModelInfo,
};
use demucs_core::provider::ModelProvider;
use demucs_core::provider::fs::FsProvider;
use demucs_core::{Demucs, ModelOptions};

/// 默认模型目录（相对仓库根解析；镜像 LocalDub `data/models/demucs`）。
pub const DEFAULT_MODEL_DIR: &str = "data/models/demucs";

#[derive(Parser)]
#[command(name = "demucs-burn")]
pub struct Cli {
    /// Benchmark mode: load model and print timing, then exit
    #[arg(long)]
    pub benchmark_load: bool,

    /// Input WAV file (for separation or load test)
    pub input: Option<PathBuf>,

    /// Output directory for stems
    pub output: Option<PathBuf>,

    /// Model variant
    #[arg(short, long, default_value = "htdemucs_ft")]
    pub model: String,

    /// 模型目录（默认仓库根 data/models/demucs，可经 DEMUCS_MODEL_DIR 覆盖）。
    /// 仓库由可执行文件位置上溯两级推断（target/… 或 data/bin 布局），不依赖 cwd。
    #[arg(long, env = "DEMUCS_MODEL_DIR", default_value = DEFAULT_MODEL_DIR)]
    pub model_dir: String,

    /// Wgpu tasks_max (CPU threads for command recording). Default 128.
    #[arg(long, default_value = "128")]
    pub tasks_max: u32,

    /// Run warmup inference to pre-compile GPU shaders.
    #[arg(long)]
    pub warmup: bool,

    /// Benchmark rounds: run separation N times in one process (default 1).
    #[arg(long, default_value = "1")]
    pub benchmark_rounds: u32,
}

/// 仓库根：二进制在 `target/{debug,release}` 或 `data/bin`，上溯两级到仓库根。
fn current_exe_repo_root() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("获取当前可执行文件路径失败")?;
    let exe_dir = exe.parent().context("可执行文件无父目录")?.to_path_buf();
    let root = exe_dir
        .join("..")
        .join("..")
        .canonicalize()
        .context("解析仓库根失败（确认从仓库内构建）")?;
    Ok(root)
}

/// 相对路径相对仓库根解析；绝对路径原样返回。
fn resolve_path(root: &Path, p: &str) -> PathBuf {
    let path = PathBuf::from(p);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

pub fn resolve_model_info(model_id: &str) -> Result<&'static ModelInfo> {
    ALL_MODELS
        .iter()
        .find(|m| m.id == model_id)
        .copied()
        .with_context(|| format!("Unknown model: {}", model_id))
}

fn model_options(info: &ModelInfo) -> ModelOptions {
    match info.id {
        HTDEMUCS_ID => ModelOptions::FourStem,
        HTDEMUCS_6S_ID => ModelOptions::SixStem,
        HTDEMUCS_FT_ID => ModelOptions::FineTuned(info.stems.to_vec()),
        _ => ModelOptions::FourStem,
    }
}

struct BenchListener;

impl ForwardListener for BenchListener {
    fn on_event(&mut self, event: ForwardEvent) {
        match event {
            ForwardEvent::ChunkStarted { index, total } => {
                let pct = (index as f64 / total as f64 * 100.0) as u32;
                println!("({}%)", pct.min(99));
            }
            ForwardEvent::ChunkDone { index, total } => {
                let pct = ((index + 1) as f64 / total as f64 * 100.0) as u32;
                println!("({}%)", pct.min(100));
            }
            _ => {}
        }
    }
}

/// 主流程。stdout 约定（separate stage 依赖，勿改）：进度 `(xx%)`、基准行 `Benchmark-*`。
pub fn run<B: Backend>(cli: Cli, device: B::Device) -> Result<()> {
    let info = resolve_model_info(&cli.model)?;
    let opts = model_options(info);

    let repo_root = current_exe_repo_root()?;
    let model_dir = resolve_path(&repo_root, &cli.model_dir);
    let provider = FsProvider::with_dir(model_dir);
    let bytes = if provider.is_cached(info) {
        eprintln!("Loading cached model: {}", info.id);
        provider
            .load_cached(info)
            .context("Failed to load cached model")?
    } else {
        anyhow::bail!(
            "Model '{}' not cached. Run demucs-cli first to download it.",
            info.id
        );
    };

    let load_start = Instant::now();
    let model =
        Demucs::<B>::from_bytes(opts, &bytes, device).context("Failed to load model weights")?;
    let load_time = load_start.elapsed();
    eprintln!("Model loaded in {:.3}s", load_time.as_secs_f64());
    println!("Benchmark-Load-Time: {:.3}", load_time.as_secs_f64());

    // 仅 wgpu/vulkan 是 GPU-shader 后端; 与旧互斥 cfg 链的 not(tch|cpu|rocm|cuda) 等价。
    if cli.warmup {
        #[cfg(any(feature = "wgpu", feature = "vulkan"))]
        {
            eprintln!("Pre-compiling GPU shaders (first run only)...");
            let warmup_start = Instant::now();
            pollster::block_on(model.warmup());
            let t = warmup_start.elapsed();
            eprintln!("Warmup done in {:.1}s", t.as_secs_f64());
            println!("Benchmark-Warmup-Time: {:.3}", t.as_secs_f64());
        }
        #[cfg(not(any(feature = "wgpu", feature = "vulkan")))]
        eprintln!("Skipping GPU warmup (non-GPU backend)");
    }

    if cli.benchmark_load {
        return Ok(());
    }

    let input = cli.input.context("Input file required")?;
    let out_dir = cli.output.context("Output directory required")?;

    eprintln!("Reading {}", input.display());
    let (left, right, sample_rate) = read_wav(&input)?;
    let duration_secs = left.len() as f64 / sample_rate as f64;
    eprintln!(
        "  {} samples, {:.1}s, {} Hz, stereo",
        left.len(),
        duration_secs,
        sample_rate
    );

    for round in 1..=cli.benchmark_rounds {
        eprintln!("--- Round {}/{} ---", round, cli.benchmark_rounds);
        let sep_start = Instant::now();
        let stems = pollster::block_on(model.separate_with_listener(
            &left,
            &right,
            sample_rate,
            &mut BenchListener,
        ))?;
        let sep_time = sep_start.elapsed();
        eprintln!("Generate (round {}): {:.3}s", round, sep_time.as_secs_f64());
        println!(
            "Benchmark-Gen-Time-Round{}: {:.3}",
            round,
            sep_time.as_secs_f64()
        );

        if round == cli.benchmark_rounds {
            std::fs::create_dir_all(&out_dir)?;
            for (i, stem) in stems.iter().enumerate() {
                let filename = format!("target_{}_{}.wav", i, stem.id.as_str());
                let path = out_dir.join(&filename);
                write_wav(&path, &stem.left, &stem.right, sample_rate)?;
                eprintln!("  Wrote {}", path.display());
            }
        }
    }

    println!("(100%)");
    eprintln!("Done!");
    Ok(())
}

pub fn read_wav(path: &PathBuf) -> Result<(Vec<f32>, Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV: {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels > 2 {
        anyhow::bail!("Expected mono or stereo, got {} channels", spec.channels);
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

pub fn write_wav(path: &std::path::Path, left: &[f32], right: &[f32], sample_rate: u32) -> Result<()> {
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
