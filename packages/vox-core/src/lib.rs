//! vox-core — 通用音频/网络基础件。
//!
//! - [`wav`]: 极简 WAV 读写 + 重采样 (16/32-bit PCM, 零音频库依赖)
//! - [`gradio_client`]: 极简 Gradio 5 HTTP 客户端 (blocking,
//!   `/gradio_api/call/{api}` 两段式 + SSE, 覆盖 LocalDub/voxlab 所需)

pub mod wav;

pub use wav::{read_wav, resample, write_wav};
