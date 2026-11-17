---
number: 10
title: GPU 路径采用 demucs-wgpu（基于上游 WGSL 实现），不深挖 ROCm 库
status: accepted
date: 2026-11-17
---

# 10. GPU 路径采用 demucs-wgpu（基于上游 WGSL 实现）

## 状态

已接受。`submodule/demucs-wgpu`（fork）与 `packages/demucs-wgpu`（封装）已接入，
CLI 契约与 `demucs-tch` 一致，输出与上游 CLI 逐样本相同（maxdiff 0.000000）。

## 背景

本机是 Radeon 780M（gfx1103 核显），此前三条 GPU 尝试都不理想：

1. **PyTorch + ROCm**：`torch 2.14.1+rocm7.2` 能识别设备，但 wheel 自带的 rocBLAS
   没有 gfx1103 的 Tensile 内核，必须 `HSA_OVERRIDE_GFX_VERSION=11.0.0` 借用
   gfx1100 的 kernel：能跑，但两次 GPU 之间只有 corr 0.95；改用系统 rocBLAS 内核
   则 RTF 3.49（比 CPU 慢 3.5 倍）。
2. **Burn wgpu（0.20.1）**：RTF 2.41，比 CPU 的 1.26 慢一倍。
3. **demucs-tch（libtorch CPU）**：RTF 1.03，是此前的可用基线。

问题不在 Demucs 的算子形状，而在通用推理栈：libtorch 依赖的 ROCm 数学库缺少
该 GPU 的内核，Burn 的 conv 实现与调度开销又吃掉了 GPU 的收益。

## 决策

- **GPU 路径 = `submodule/demucs-wgpu`**（上游 `eclipse005/demucs-wgpu` 的 WGSL kernel + wgpu，Vulkan 已验证）。
  它是完整的 HTDemucs 前向实现（conv / attention / STFT / overlap-add 全自研），
  不依赖 libtorch、ROCm、CUDA。
- **`packages/demucs-wgpu` 只做薄封装**：解析权重路径、剥掉 safetensors 的
  signature 前缀、按 `demucs-tch` 的 stdout 契约输出、写 `target_<i>_<stem>.wav`。
- **不再深挖 ROCm/MIOpen**：那是替通用框架补内核覆盖，补完也只有 RTF ~0.6；
  也不必为 Burn 写自定义 conv。
- **默认 `shifts = 0`**（确定性单趟），与基准/数值对比口径一致；上游
  `SeparateOptions::default()` 是 1（随机时移平均）。

## 实测（同一段 33.878 s 音轨，`shifts=0`）

| 路径 | 模型 | 时间 | RTF |
|------|------|------|-----|
| demucs-tch（libtorch CPU） | htdemucs | 35.05 s | 1.03 |
| demucs-tch（libtorch CPU） | htdemucs_ft | 160.53 s | 4.74 |
| PyTorch ROCm（arch override） | htdemucs | 21.78 s | 0.64 |
| demucs-wgpu CPU（自研 host 路径） | htdemucs | 46.05 s | 1.36 |
| **demucs-wgpu Vulkan** | htdemucs | **3.61 s** | **0.11** |
| **demucs-wgpu Vulkan** | htdemucs_ft | **15.3 s** | **0.45** |

RTF = 处理时间 / 音频时长（越小越快）。注意 demucs-wgpu README 用的是 RTFx（倒数）。

数值：Vulkan 与它自己的 CPU 后端 corr 1.0000（SDR 103–119 dB）；与 `demucs-tch`
对比 drums 0.9979 / other 0.9955 / vocals 0.9989（bass 0.7492，弱 stem 的已知差异）。

## 后果

- 权重布局与 `demucs-tch` 不同：`htdemucs_ft` 必须是**目录**
  （`htdemucs_ft.yaml` + 四个 `<signature>.safetensors`），不能是合并的单一文件。
- `packages/demucs-wgpu` 是独立 workspace（root `Cargo.toml` 的 `exclude`），
  自带 `Cargo.lock`；构建产物在 `packages/demucs-wgpu/target/`。
- 加载时会把带 `955717e8.` 之类前缀的 safetensors 规范化到临时目录缓存，
  不改上游代码，也不改原始权重文件。
- 上游范围限制：只支持 `htdemucs` / `htdemucs_ft`，不含 6s、不含 mp3/flac。
- 若换到 NVIDIA 独显，CUDA + PyTorch 仍是更快/更省事的选择；本决策针对的是
  「通用栈缺内核」这类情况，不是否定 CUDA 路径。

## 相关

- [0008](./0008-demucs-tch-standalone-workspace.md) demucs-tch 独立 workspace
- [0009](./0009-demucs-tch-backend-builds.md) demucs-tch 按后端构建
- `submodule/demucs-wgpu`、`packages/demucs-wgpu`
- `packages/demucs-burn/docs/gpu-backend-outlook.md`（此前的 GPU 探索记录）
