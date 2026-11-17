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

## 实验环境记录（2026-11-17，未完成）

参考口径：用户线上 **RTF 4–6**（生成耗时 / 音频时长，越小越快），且是开启
`torch.compile`（`optimize=True`）后测的。我们默认 `optimize=False` 是为了让
profiler 看到真实算子耗时，两者不可直接对比。

实验最终在 **Python 3.12 + CPU torch 2.14.1** 环境下跑通（`.venv312`）。`.venv`
（3.14 + torch 2.14.1+rocm7.2）上跑不通，卡点记录如下（避免重复踩）：

- `torchaudio` 无 cp314 wheel（PyPI 与 rocm7.2 索引都没有），`voxcpm` 顶层
  无条件 import → `core` 里加了桩，音频 IO 路径会显式报错。
- 模型能加载完（日志到 `Running on device: ...`），但生成阶段在约 26 s 后收到
  SIGTERM 退出，CPU / GPU 均如此；`--max-len 8`、`--dit-steps 2` 也无法绕过，
  说明不是「慢」而是环境问题（python 3.14 + torch 2.14+rocm7.2 + 桩的组合）。
- 因此分段耗时与 dtype SNR 尚未获得；工具本身（`core` / `cli` / `server`）已就绪。

换到 Python 3.12 + CPU torch 后一切正常，说明是 3.14 / ROCm 版 torch / 桩的组合
问题，不是模型或代码问题。

## 实测：分段耗时（2026-11-17）

环境：`.venv312`（Python 3.12 + `torch 2.14.1+cpu`），`--device cpu --max-len 32
--dit-steps 4`，`optimize=False`，墙钟 23.86 s / 音频 2.88 s → **RTF 8.29**。

计时方式：对 `tts_model` 的**叶子模块**注册 forward hook，按顶层模块名归组
（容器可能通过子模块调用，例如 `audio_vae.decoder(x)`，只 hook 顶层会漏掉大块）。

| 阶段 | 耗时 | 占比 |
|------|------|------|
| `base_lm`（MiniCPM4，1472M） | 4793 ms | 36.2% |
| `feat_decoder` = UnifiedCFM（212M，DiT/CFM） | 3921 ms | 29.6% |
| `audio_vae`（94M） | 1794 ms | 13.6% |
| `residual_lm`（378M） | 1370 ms | 10.4% |
| `feat_encoder`（208M） | 1249 ms | 9.4% |
| 投影层合计 | ~100 ms | 0.8% |
| **未归因**（Python 循环、KV cache、采样） | 10633 ms | 44.6% |

**结论修正**：最初假设「LM 解码是绝对大头，GPU 救不了」。实测 **LM 合计 46.6%**，
而 **CFM 29.6% + VAE 13.6% + encoder 9.4% = 52.6%**，这三段都是并行、批量、
卷积/注意力密集的形状。因此可行的路线不是全栈重写，而是**只把 CFM + VAE 搬到
GPU**（kernel 骨架可复用 demucs-wgpu），LM 留在 CPU。

未完成：`dtype` 实验（bf16 vs fp32）。fp32 权重 8.7 GB，本机 27 GB 内存此时已用
16 GB、`/proc/pressure/memory` 显示换页，fp32 生成无法在合理时间内完成。
建议改在有显存余量的机器上做，或先用 `--max-len 8 --dit-steps 2`。

## Burn 路径实测（2026-11-17）

用 `submodule/voxcpm-rs` 的 `examples/tts`（`VOXCPM_TIMESTEPS=4`，模型同为
`data/models/voxcpm2`，文本“你好。”，产出 0.80 s 音频）：

| 后端 | 加载 | 生成 | RTF |
|------|------|------|-----|
| Burn `cpu`（ndarray + simd） | 65.7 s | 270.5 s | **338.1** |
| Burn `cpu-blas`（OpenBLAS 0.3.34） | 71.6 s | 314.2 s | **392.7** |
| PyTorch CPU（bf16，同机） | 29.4 s | 23.9 s | **8.29** |

结论：Burn 的 CPU 路径比 PyTorch 慢约 40 倍，且 `cpu-blas` 没有带来改善 ——
说明瓶颈不在 matmul（Burn 侧是 f32 权重 + 朴素 conv），而在卷积实现与
缺少 bf16 支持。CPU 上 Burn 不是 VoxCPM2 的可用路径。

Vulkan 路径当时无法验证（patch 只在 workspace 根生效），后已通过把 voxcpm-rs
改为独立 workspace 解决，结果见下节。

## 结构变更（2026-11-17）：voxcpm-rs 改为独立 workspace

`submodule/voxcpm-rs` 不再作为 vox-lab 的 workspace 成员，改为**独立 workspace 根**，
原因是 bf16/SPIR-V 依赖的 `[patch.crates-io]`（`vendor/patched/{burn-cubecl,cubecl-spirv}`）
只在 workspace 根生效。原 `packages/voxcpm-burn` 随之移入
`submodule/voxcpm-rs/vox-lab-burn/`（crate 名 `vox-lab-burn`，lib 名仍为
`voxcpm_burn`），三个后端 bin 在子模块内构建。

## Burn 的 Vulkan / wgpu 实测（2026-11-17，780M）

| 后端 | 结果 |
|------|------|
| `vulkan`（bf16） | 加载 58.9 s，生成时 panic：`cubek-matmul` 报 `BF16 lhs/rhs/output not supported` —— 与 [0010](./0010-demucs-wgpu-backend.md) 的结论一致，gfx1103 没有 bf16 matmul |
| `wgpu`（f32） | 加载 96.9 s，生成时 `radv/amdgpu: Not enough memory for command submission`，随后 wgpu 校验失败 panic |
| `cpu`（ndarray） | RTF 338.1 |
| `cpu-blas`（OpenBLAS） | RTF 392.7 |

wgpu f32 失败的直接原因：4.4 GB bf16 权重在 f32 下是 **8.7 GB**，本机是核显，
显存来自共享内存（27 GB 总量，此时已用约 16 GB），放不下权重 + 激活 + workspace。
这不是 shader 问题，是容量问题。

## 相关

- [0010](./0010-demucs-wgpu-backend.md) demucs 的自研 WGSL 路径
- [0009](./0009-demucs-tch-backend-builds.md) tch 的后端构建
- `packages/voxcpm_pytorch/justfile`
