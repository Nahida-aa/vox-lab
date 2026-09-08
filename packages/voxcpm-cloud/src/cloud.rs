//! VoxCPM cloud (Gradio) TTS backend.
//!
//! 使用 [`gradio`] crate（Gradio 官方推荐的 Rust 客户端，支持 Gradio 4/5/6）
//! 调用 `/generate`。TS 参考实现见 `src/index.ts`
//! （契约实测自 https://voxcpm.modelbest.cn，Gradio 6.10）。

use anyhow::anyhow;
use gradio::{Client, ClientOptions, PredictionInput, PredictionOutput};

use vox_core::wav;

pub const DEFAULT_API_URL: &str = "https://voxcpm.modelbest.cn";

/// Output sample rate we normalize everything to (matches TS cloud.ts).
pub const TARGET_SAMPLE_RATE: u32 = 48000;

#[derive(Debug, Clone, Default)]
pub struct VoxCPMCloudConfig {
    /// Base URL of the Gradio server (no trailing slash). Defaults to [`DEFAULT_API_URL`].
    pub api_url: Option<String>,
    /// Optional control instruction for text-only synthesis.
    pub control_instruction: Option<String>,
}

#[derive(Debug)]
pub struct VoxCPMCloud {
    client: Client,
    control_instruction: String,
}

/// Result of a single TTS generation: raw PCM f32 samples normalized to [`TARGET_SAMPLE_RATE`].
#[derive(Debug)]
pub struct TtsResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub gen_time_sec: f64,
}

impl VoxCPMCloud {
    pub fn new(config: VoxCPMCloudConfig) -> anyhow::Result<Self> {
        let base_url = config
            .api_url
            .unwrap_or_else(|| DEFAULT_API_URL.to_string());
        // `new_sync` 会拉取 /config 与 /info (由 crate 内部完成), 直传 URL 即可。
        let client = Client::new_sync(&base_url, ClientOptions::default())?;
        Ok(Self {
            client,
            control_instruction: config.control_instruction.unwrap_or_default(),
        })
    }

    /// Generate speech from `text` using a local reference WAV file (`reference_wav_path`).
    /// `prompt_text` is the original-language transcription of the reference audio (improves
    /// cloning quality); pass empty/None when unavailable.
    pub fn generate(
        &self,
        text: &str,
        reference_wav_path: &str,
        prompt_text: Option<&str>,
        cfg_value: f64,
    ) -> anyhow::Result<TtsResult> {
        let t_start = std::time::Instant::now();

        // 位置参数顺序 (与 TS index.ts / Gradio 6.10 契约一致):
        //   text, control_instruction, ref_wav, use_prompt_text, prompt_text_value,
        //   cfg_value, do_normalize, denoise, dit_steps, user_id
        let inputs = vec![
            PredictionInput::from_value(text),
            PredictionInput::from_value(&self.control_instruction),
            PredictionInput::from_file(reference_wav_path),
            PredictionInput::from_value(false),
            PredictionInput::from_value(prompt_text.unwrap_or("")),
            PredictionInput::from_value(cfg_value),
            PredictionInput::from_value(false),
            PredictionInput::from_value(false),
            PredictionInput::from_value(10),
            PredictionInput::from_value(""),
        ];

        let outputs = self.client.predict_sync("/generate", inputs)?;

        let audio = outputs
            .into_iter()
            .find_map(|o| match o {
                PredictionOutput::File(f) => Some(f),
                _ => None,
            })
            .ok_or_else(|| anyhow!("generate 返回无音频 FileData"))?;

        let wav_bytes = audio.download_sync(None)?;
        let (samples, sample_rate) = wav::read_wav(&wav_bytes)?;
        let (samples, sample_rate) = if sample_rate != TARGET_SAMPLE_RATE {
            (
                wav::resample(&samples, sample_rate, TARGET_SAMPLE_RATE),
                TARGET_SAMPLE_RATE,
            )
        } else {
            (samples, sample_rate)
        };

        Ok(TtsResult {
            samples,
            sample_rate,
            gen_time_sec: t_start.elapsed().as_secs_f64(),
        })
    }
}
