//! cloud TTS 冒烟测试 (example, 不进正式构建产物)。
//!
//! ```bash
//! cargo run -p voxcpm-cloud --example smoke -- --ref <reference.wav> --text "..."
//! ```
//!
//! 参数说明见 `--help`。

use std::path::PathBuf;

use clap::Parser;
use vox_core::write_wav;
use voxcpm_cloud::{VoxCPMCloud, VoxCPMCloudConfig};

#[derive(Parser, Debug)]
#[command(name = "voxcpm-cloud-smoke", about = "VoxCPM cloud (Gradio) TTS 冒烟测试")]
struct Args {
    /// 参考音频路径 (声音克隆必需)
    #[arg(long = "ref", value_name = "WAV")]
    reference: PathBuf,

    /// 合成文本
    #[arg(long, default_value = "好的，我们现在在大象面前。")]
    text: String,

    /// 输出 wav 路径
    #[arg(long, default_value = "tmp/voxcpm-cloud-out.wav")]
    out: PathBuf,

    /// Gradio 服务 URL (默认 https://voxcpm.modelbest.cn)
    #[arg(long)]
    api_url: Option<String>,

    /// 参考音频的转写文本 (提升克隆质量)
    #[arg(long)]
    prompt: Option<String>,

    /// CFG 值
    #[arg(long, default_value_t = 2.0)]
    cfg: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let cloud = VoxCPMCloud::new(VoxCPMCloudConfig {
        api_url: args.api_url,
        control_instruction: None,
    })?;

    println!("[voxcpm-cloud] generating via cloud backend...");
    let result = cloud.generate(
        &args.text,
        &args.reference.to_string_lossy(),
        args.prompt.as_deref(),
        args.cfg,
    )?;

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_wav(
        &result.samples,
        result.sample_rate,
        &args.out.to_string_lossy(),
    )?;

    let dur = result.samples.len() as f64 / result.sample_rate as f64;
    println!(
        "[voxcpm-cloud] wrote {:.2}s @ {}Hz to {} (gen {:.2}s)",
        dur,
        result.sample_rate,
        args.out.display(),
        result.gen_time_sec
    );
    Ok(())
}
