---
number: 11
title: VoxCPM 的 GPU 路径先做量化实验（dtype / 分段耗时），不做全栈 wgpu 运行时
status: proposed
date: 2026-11-17
---

# 11. VoxCPM 的 GPU 路径：先量化，再决定是否自研

## 状态

提出。实验工具已落地（`packages/voxcpm_pytorch` 的 `core` / `cli`），结论待测。

## 背景

demucs 在 780M 上靠自研 WGSL kernel 拿到 RTF 0.11（见 [0010](./0010-demucs-wgpu-backend.md)），
自然会问 VoxCPM 能否照做。两者结构差别很大：

| | demucs | VoxCPM2 |
|---|---|---|
| 权重 | 84 MB fp32 ×4 | **4.4 GB BF16**（约 2.18B 参数，单文件 577 张量） |
| 计算形态 | 固定 segment 的卷积 + transformer，一次前向出全部 stem | 自回归：MiniCPM4 LM（28 层 / hidden 2048 / GQA 2 kv）逐 patch 解码 + 残差 LM（8 层）+ DiT（12 层）× 10 步 CFM + AudioVAE 解码 |
| 形状 | 静态，可特化 | 序列长度逐 token 增长，KV cache 滚动 |
| 瓶颈 | FLOPs | **权重带宽**（每个 token 都要重读大部分权重） |

bf16 支持现状（本机核对 2026-11-17）：

| 栈 | 结论 |
|---|---|
| wgpu / WGSL | 只有 `Features::SHADER_F16`；`wgpu-types 30` 无 bf16 特性 |
| 原生 Vulkan | `VK_KHR_shader_bfloat16` 规范存在，但本机 RADV/Mesa 26.2.7 + gfx1103 未暴露；仅有 `shaderFloat16 = true` |
| ROCm / HIP | 原生 bf16（`hip_bfloat16.h`、gfx1103 有 bf16 指令）；障碍在 rocBLAS/MIOpen 缺 gfx1103 内核 |

因此 bf16 不是「不适合 GPU」的判决，而是「要转 f16（范围风险）、上 f32（8.7 GB、带宽翻倍）
或在 shader 里用 u16 手工承载」的额外成本。真正决定收益的是自回归 + 带宽受限：
核显与 CPU 共用 LPDDR5（约 100–120 GB/s），每 token 读一遍 4.4 GB 权重量级就要 40 ms，
而 demucs 的加速来自算力密集 + 批量 + 静态形状。

## 决策

1. **先量化，不先写 kernel**。两条数据决定后续：
   - `cli profile`：LM 解码 / 残差 LM / CFM / AudioVAE 各占多少时间
   - `cli dtype`：bf16 vs fp32（可选 fp16）的 SNR 与耗时差
2. **`packages/voxcpm_torch_server` 改名 `voxcpm_pytorch` 并拆模块**：
   - `core.py`：加载 / 生成 / 计时 / profiler 聚合 / dtype 实验（无 HTTP 依赖）
   - `cli.py`：`generate` / `profile` / `dtype` 三个子命令
   - `server.py`：保留复刻云端 `/generate` 的 Gradio + FastAPI 服务，仅编排 `core`
   （与 `demucs_pytorch` 的模块划分保持一致）
3. **默认 `optimize=False`**：`torch.compile` 会掩盖单算子成本、破坏 profiling；
   只在吞吐测量时开启。
4. **不做全栈 wgpu 运行时**，直到上面两条数据显示 CFM/VAE 占比足够高（即只 GPU 化
   这两段就能拿到主要收益）。

## 后果

- uv workspace member 由 `packages/voxcpm_torch_server` 改为 `packages/voxcpm_pytorch`；
  README / ROADMAP / ADR 0003、0004、0006 的引用同步更新。
- 服务启动方式改为 `python -m voxcpm_pytorch.server`，新增 `--load-on-start`。
- `core.dtype_experiment` 一次只持有一份模型（bf16 4.4 GB / f32 8.7 GB），
  逐份加载并 `gc`，避免同时驻留两份。
- 若实验显示 LM 解码占绝对多数，则结论是「GPU 救不了」，继续用 CPU；
  若 CFM/VAE 占比高，再评估只搬这两段（可复用 demucs-wgpu 的 kernel 骨架）。

## 相关

- [0010](./0010-demucs-wgpu-backend.md) demucs 的自研 WGSL 路径
- [0009](./0009-demucs-tch-backend-builds.md) tch 的后端构建
- `packages/voxcpm_pytorch/justfile`
