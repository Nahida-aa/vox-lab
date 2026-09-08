//! 异步 Client::new 验证: HF Space (6.0.0) 与官方站 (6.10.0) 各走一遍
//! new + predict, 确认 async 路径与 label i18n 修复的行为一致。

use gradio::{Client, ClientOptions, PredictionInput, PredictionOutput};

#[gradio::tokio::main]
async fn main() -> anyhow::Result<()> {
    let ref_wav = "/home/aa/repos/env_ls/LocalDub/workfolder/大/93/tts/wavs/0001.wav";
    let cases: &[(&str, &str)] = &[
        ("https://openbmb-voxcpm-demo.hf.space", "HF Space (6.0.0)"),
        ("https://voxcpm.modelbest.cn", "official (6.10.0)"),
    ];
    for (url, label) in cases {
        println!("=== async new: {label} ===");
        let client = match Client::new(url, ClientOptions::default()).await {
            Ok(c) => {
                println!("  new ok");
                c
            }
            Err(e) => {
                println!("  new FAILED: {e}");
                continue;
            }
        };
        let inputs = vec![
            PredictionInput::from_value("异步路径验证。"),
            PredictionInput::from_value(""),
            PredictionInput::from_file(ref_wav),
            PredictionInput::from_value(false),
            PredictionInput::from_value(""),
            PredictionInput::from_value(2.0),
            PredictionInput::from_value(false),
            PredictionInput::from_value(false),
            PredictionInput::from_value(10),
            PredictionInput::from_value(""),
        ];
        match client.predict("/generate", inputs).await {
            Ok(outputs) => {
                let url_out = outputs.iter().find_map(|o| match o {
                    PredictionOutput::File(f) => f.url.clone(),
                    _ => None,
                });
                println!(
                    "  predict ok: {} outputs, file_url={}",
                    outputs.len(),
                    url_out.is_some()
                );
            }
            Err(e) => println!("  predict FAILED: {e}"),
        }
    }
    Ok(())
}
