---
number: 6
title: 子模块与运行时依赖选型
status: accepted
date: 2026-11-14
---

# 6. 子模块与运行时依赖选型

## 状态

已接受。

## 决策

以 `submodule/voxcpm-rs`（纯 Rust / Burn 的 voxcpm2 推理，cpu / vulkan / wgpu）
作为 Burn 侧的基础依赖与参考；它已提供 `VoxCPM::from_local(dir, &device)` +
`generate(text, opts)`，按 feature 切 Burn 后端。

以 `submodule/VoxCPM`（官方 PyTorch 实现，`torch>=2.5`）作为 PyTorch 侧的参考实现，
推理入口在 `src/voxcpm/core.py` 与 `cli.py`。本地 `voxcpm_pytorch` 用
`VoxCPM.from_pretrained` 在 cpu / cuda 加载权重。

| 子模块 | 链路 |
|--------|------|
| `voxcpm-rs` | TTS / Burn |
| `VoxCPM` | TTS / PyTorch 参考 |
| `demucs-rs` | 分离 / Rust |
| `demucs`（adefossez） | 分离 / 权重与参考 |
| `demucs.cpp` | 分离 / GGML 基准 |
| `whisper.cpp`（钉在 LocalDub 同 commit，Vulkan 后端） | ASR |
| `CosyVoice`（CosyVoice2/3） | TTS / VC 基准 |

## 未落地的早期命名

立项规划中的 `runtime-burn` / `runtime-pytorch` / `runtime-cloud` / `shared-proto` /
`cli` / `ts-client` / `config` 等 crate **均未按该命名落地**。实际能力分散在：

- Rust：`voxcpm-burn`、`demucs-burn`、`voxcpm-cloud`、`vox-core`、`voice-cloning`、`util`
- Python：`voxcpm_pytorch`、`demucs_pytorch`（PyTorch 参考验证工具）、`servers_py`

详见 README「仓库布局（现状）」与 [0003](./0003-rust-cli-python-http.md)。

## 后果

- 新增实现优先作为独立 crate 落在 `packages/` 下，而非复活早期命名。
- `submodule/whisper.cpp` 的 commit 与 LocalDub 对齐，升级需两侧同步，避免 CI 发布资产漂移。
