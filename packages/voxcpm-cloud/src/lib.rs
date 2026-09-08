//! voxcpm-cloud — VoxCPM cloud (Gradio) TTS 后端。
//!
//! Rust 移植自 LocalDub `packages/voxlab/src/engines/voxcpm/cloud.rs`,
//! TS 参考实现见 `src/index.ts` (契约实测自 https://voxcpm.modelbest.cn)。

mod cloud;

pub use cloud::{DEFAULT_API_URL, TARGET_SAMPLE_RATE, TtsResult, VoxCPMCloud, VoxCPMCloudConfig};
