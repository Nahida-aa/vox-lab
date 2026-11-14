//! vox-core — 通用音频基础件。
//!
//! - [`wav`]: 极简 WAV 读写 + 重采样 (16/32-bit PCM, 零音频库依赖)

pub mod wav;

pub use wav::{read_wav, resample, write_wav};
