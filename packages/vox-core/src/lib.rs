//! vox-core — 通用音频基础件。
//!
//! - [\`wav\`]: 极简 WAV 读写 + 重采样 (16/32-bit PCM, 零音频库依赖)
//!
//! Gradio 客户端改用官方推荐的 \`gradio\` crate, 由 \`voxcpm-cloud\` 直接依赖。

pub mod wav;

pub use wav::{read_wav, resample, write_wav};
