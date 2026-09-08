//! HF Space 验证: 用 packages/gradio (0.4.1 + 6.10 label 兼容) 调
//! openbmb-voxcpm-demo.hf.space (Gradio 6.0.0) 与 voxcpm.modelbest.cn (6.10.0),
//! 确认 label i18n 修复有效。
use gradio::{Client, ClientOptions, PredictionInput};

fn main() -> anyhow::Result<()> {
    let ref_wav =
        "/home/aa/repos/env_ls/LocalDub/workfolder/大/93/tts/wavs/0001.wav";
    let cases: &[(&str, &str)] = &[
        ("https://openbmb-voxcpm-demo.hf.space", "HF Space (6.0.0)"),
        ("https://voxcpm.modelbest.cn", "official (6.10.0)"),
    ];
    for (url, label) in cases {
        println!("=== {label}: {url} ===");
        let client = match Client::new_sync(url, ClientOptions::default()) {
            Ok(c) => {
                println!("  new_sync ok");
                c
            }
            Err(e) => {
                println!("  new_sync FAILED: {e}");
                continue;
            }
        };
        let inputs = vec![
            PredictionInput::from_value("验证 label 兼容。"),
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
        match client.predict_sync("/generate", inputs) {
            Ok(out) => {
                let url = out.iter().find_map(|o| match o {
                    gradio::PredictionOutput::File(f) => f.url.clone(),
                    _ => None,
                });
                println!("  predict ok: {} outputs, url={:?}", out.len(), url.is_some());
            }
            Err(e) => println!("  predict FAILED: {e}"),
        }
    }
    Ok(())
}
