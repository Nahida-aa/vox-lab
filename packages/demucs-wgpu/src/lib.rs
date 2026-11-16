//! demucs-wgpu — vox-lab 的薄封装：把 `submodule/demucs-wgpu` 的 wgpu 推理接成
//! 与 `demucs-tch` 相同的 CLI 契约（同样的 stdout 行、同样的 stem 文件命名）。
//!
//! 与 tch 版的关键差别：这里没有 libtorch，也没有 rocBLAS/MIOpen。HTDemucs 的
//! conv / attention 全部由 WGSL kernel 实现，因此不受 ROCm 缺少 gfx1103 内核的影响。
//!
//! 另外上游 checkpoint 有两种键名布局：官方 `.th` / HuggingFace 快照用裸键
//! （`encoder.0.conv.weight`），而 set-soft 的 safetensors 带 signature 前缀
//! （`955717e8.encoder.0.conv.weight`）。加载前统一剥掉前缀并缓存到临时目录，
//! 这样不改上游代码也能读我们已有的权重文件。

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use demucs_core::demucs::pipeline::SeparateOptions;
use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection};

pub const DEFAULT_MODEL_DIR: &str = "data/models/demucs";

#[derive(Parser, Debug)]
#[command(name = "demucs-wgpu")]
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

    /// No-op for the wgpu backend (kept for CLI compatibility).
    #[arg(long, default_value = "128")]
    pub tasks_max: u32,

    /// Accepted for CLI compatibility; wgpu pipelines are built at load time.
    #[arg(long)]
    pub warmup: bool,

    /// Number of in-process inference rounds.
    #[arg(long, default_value = "1")]
    pub benchmark_rounds: u32,
}

// ---------------------------------------------------------------------------
// Path resolution (mirrors demucs-tch/src/lib.rs)
// ---------------------------------------------------------------------------

pub fn current_exe_repo_root() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("current_exe failed")?;
    let mut dir = exe.parent().context("could not resolve exe parent")?.to_path_buf();
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

fn variant_of(name: &str) -> Result<ModelVariant> {
    match name {
        "htdemucs" => Ok(ModelVariant::FourStem),
        "htdemucs_ft" => Ok(ModelVariant::FineTuned),
        other => bail!("unknown model '{}' (expected htdemucs or htdemucs_ft)", other),
    }
}

/// Resolve the checkpoint this variant needs.
///
/// `htdemucs`      → `<model_dir>/htdemucs.safetensors` or `htdemucs.th`
/// `htdemucs_ft`   → `<model_dir>/htdemucs_ft/`（内含 yaml + 四个 shard）
fn checkpoint_path(variant: ModelVariant, model_dir: &Path) -> Result<PathBuf> {
    match variant {
        ModelVariant::FourStem => {
            let st = model_dir.join("htdemucs.safetensors");
            if st.is_file() {
                return Ok(st);
            }
            let th = model_dir.join("htdemucs.th");
            if th.is_file() {
                return Ok(th);
            }
            bail!(
                "htdemucs checkpoint not found in {} (expected htdemucs.safetensors or htdemucs.th)",
                model_dir.display()
            )
        }
        ModelVariant::FineTuned => {
            let dir = model_dir.join("htdemucs_ft");
            if dir.is_dir() {
                Ok(dir)
            } else {
                bail!(
                    "htdemucs_ft bag not found at {} (expected a directory with \
                     htdemucs_ft.yaml and f7e0c4bc/d12395a8/92cfc3b6/04573f0d.safetensors)",
                    dir.display()
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint normalisation (strip the 8-hex signature prefix)
// ---------------------------------------------------------------------------

fn cache_root() -> PathBuf {
    std::env::temp_dir().join("vox-lab-demucs-wgpu")
}

/// Strip `<8 hex>.` from safetensors keys. Returns `None` when no key carries
/// such a prefix, so callers can use the original file untouched.
fn strip_signature_prefix(bytes: &[u8]) -> Result<Option<Vec<u8>>> {
    if bytes.len() < 8 {
        bail!("checkpoint is shorter than a safetensors header");
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if 8 + header_len > bytes.len() {
        bail!("safetensors header length exceeds file size");
    }
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).context("invalid safetensors header")?;
    let map = header.as_object().context("safetensors header is not an object")?;

    let mut renamed = serde_json::Map::new();
    let mut changed = false;
    for (key, value) in map {
        if key == "__metadata__" {
            renamed.insert(key.clone(), value.clone());
            continue;
        }
        let stripped = key.split_once('.').and_then(|(prefix, rest)| {
            let is_sig = prefix.len() == 8 && prefix.chars().all(|c| c.is_ascii_hexdigit());
            is_sig.then(|| rest.to_string())
        });
        match stripped {
            Some(rest) => {
                changed = true;
                renamed.insert(rest, value.clone());
            }
            None => {
                renamed.insert(key.clone(), value.clone());
            }
        }
    }
    if !changed {
        return Ok(None);
    }

    let mut new_header = serde_json::to_vec(&serde_json::Value::Object(renamed))?;
    while new_header.len() % 8 != 0 {
        new_header.push(b' ');
    }
    let mut out = Vec::with_capacity(bytes.len() + new_header.len());
    out.extend_from_slice(&(new_header.len() as u64).to_le_bytes());
    out.extend_from_slice(&new_header);
    out.extend_from_slice(&bytes[8 + header_len..]);
    Ok(Some(out))
}

fn normalize_file(src: &Path, dst: &Path) -> Result<()> {
    let bytes = std::fs::read(src)
        .with_context(|| format!("reading checkpoint {}", src.display()))?;
    let out = match strip_signature_prefix(&bytes)? {
        Some(v) => v,
        None => bytes,
    };
    if dst.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(dst, out).with_context(|| format!("writing {}", dst.display()))?;
    Ok(())
}

/// Return a path the upstream loader can read as-is.
///
/// Files are copied into a temp cache; bag directories get a mirror dir holding
/// the yaml plus normalised shards.
fn prepare_checkpoint(path: &Path) -> Result<PathBuf> {
    let root = cache_root();
    if path.is_file() {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("ckpt");
        let len = std::fs::metadata(path)?.len();
        let dst = root.join("single").join(format!("{}-{}.safetensors", name, len));
        normalize_file(path, &dst)?;
        return Ok(dst);
    }
    if !path.is_dir() {
        bail!("checkpoint path does not exist: {}", path.display());
    }

    let bag_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("bag");
    let dst_dir = root.join("bag").join(bag_name);
    std::fs::create_dir_all(&dst_dir)?;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let src = entry.path();
        let file_name = match src.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let dst = dst_dir.join(&file_name);
        match src.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => normalize_file(&src, &dst)?,
            // yaml and anything else is copied verbatim
            _ => {
                if !dst.exists() {
                    std::fs::copy(&src, &dst)?;
                }
            }
        }
    }
    Ok(dst_dir)
}

// ---------------------------------------------------------------------------
// WAV I/O (same format as demucs-tch)
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
            reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
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

/// Main entry. stdout contract mirrors `demucs-tch`: `(NN%)` progress,
/// `Benchmark-Load-Time: {:.3}`, `Benchmark-Warmup-Time: {:.3}`,
/// `Benchmark-Gen-Time-Round{N}: {:.3}` and a bare `Benchmark-Gen-Time: {:.3}`.
pub fn run(cli: Cli) -> Result<()> {
    if cli.benchmark_rounds == 0 {
        bail!("--benchmark-rounds must be >= 1");
    }
    let variant = variant_of(&cli.model)?;

    let root = current_exe_repo_root()?;
    let model_dir = resolve_path(&root, &cli.model_dir);
    let ckpt = checkpoint_path(variant, &model_dir)?;
    eprintln!("Loading checkpoint: {}", ckpt.display());

    let t_load = Instant::now();
    let prepared = prepare_checkpoint(&ckpt)?;
    let opts = LoadOptions {
        variant,
        stems: StemSelection::All,
    };
    let demucs = Demucs::load(&prepared, opts, Backend::Auto)
        .map_err(|e| anyhow::anyhow!("model load failed: {e}"))?;
    let load_secs = t_load.elapsed().as_secs_f64();
    eprintln!("Backend: {}", demucs.backend_tag());
    println!("Benchmark-Load-Time: {:.3}", load_secs);

    if cli.warmup {
        eprintln!("No explicit warmup for wgpu (pipelines are built during load)");
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

    // Deterministic by default: one fixed pass, no random shift averaging.
    let options = SeparateOptions {
        shifts: 0,
        ..Default::default()
    };

    let mut last_stems = None;
    for round in 1..=cli.benchmark_rounds {
        eprintln!("--- Round {}/{} ---", round, cli.benchmark_rounds);
        let t0 = Instant::now();
        let stems = demucs
            .separate_with_options(&left, &right, sample_rate, &options, &mut |p| {
                println!("({}%)", p.percent());
            })
            .map_err(|e| anyhow::anyhow!("separation failed: {e}"))?;
        let gen_secs = t0.elapsed().as_secs_f64();
        println!("Benchmark-Gen-Time-Round{}: {:.3}", round, gen_secs);
        if round == cli.benchmark_rounds {
            println!("Benchmark-Gen-Time: {:.3}", gen_secs);
            last_stems = Some(stems);
        }
    }

    let stems = last_stems.context("no inference round produced stems")?;
    for (i, stem) in stems.iter().enumerate() {
        let path = output.join(format!("target_{}_{}.wav", i, stem.id.as_str()));
        write_wav(&path, &stem.left, &stem.right, sample_rate)?;
        eprintln!("Wrote {}", path.display());
    }
    eprintln!("Done!");
    Ok(())
}
