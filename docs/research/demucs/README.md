# Demucs 研究记录

音源分离链路（htdemucs / htdemucs_ft）的实测与结论。决策的「为什么」在
[`docs/decisions/`](../../decisions/README.md)，这里记「测了什么、结果如何」。

相关 ADR：[0009](../../decisions/0009-demucs-tch-backend-builds.md)（tch 后端构建）、
[0010](../../decisions/0010-demucs-wgpu-backend.md)（自研 WGSL 路径）。

## 环境

- CPU：Ryzen 7 7840HS（16 线程）
- GPU：Radeon 780M（gfx1103，RADV，核显，显存与系统内存共享）
- 系统 ROCm 10.1 / hip 7.16
- 基准输入：`packages/benchmark/ref/video_source.mp4` 的音轨，33.878 s @ 44.1 kHz

## 后端对比（2026-11）

RTF = 处理时间 / 音频时长（越小越快）。`shifts=0`（确定性单趟），htdemucs 单模型。

| 路径 | 实现 | 时间 | RTF | 备注 |
|------|------|------|-----|------|
| `demucs-tch` | tch-rs / libtorch CPU | 35.05 s | 1.03 | 基线；ft 为 160.5 s（RTF 4.74） |
| PyTorch | CPU（官方 demucs） | 33.45 s | 0.99 | |
| PyTorch | ROCm 2.14.1+rocm7.2 | 21.78 s | 0.64 | 需 `HSA_OVERRIDE_GFX_VERSION=11.0.0`；两次运行结果不一致 |
| Burn（demucs-rs） | wgpu 0.20.1 | — | 2.41 | 历史数据，见 `packages/research/separate/readme.md` |
| `demucs-wgpu` | wgpu（自研 WGSL）CPU | 46.05 s | 1.36 | 同一份代码的对照 |
| **`demucs-wgpu`** | **wgpu（自研 WGSL）Vulkan** | **3.61 s** | **0.11** | ft 为 15.3 s（RTF 0.45） |

## 关键发现

### 1. 780M 上 PyTorch/ROCm 路径不可用

`torch 2.14.1+rocm7.2` 能识别设备，但 wheel 自带的 rocBLAS **没有 gfx1103 的
Tensile 内核**：

```
rocBLAS error: Cannot read .../rocblas/library/TensileLibrary.dat:
Illegal seek for GPU arch : gfx1103
```

可用的只有 gfx1030/1100/1101/1102/1150/1151/1200/1201/908/90a/942/950。
两条绕法都不理想：

| 绕法 | 时间 | RTF | 问题 |
|------|------|-----|------|
| `HSA_OVERRIDE_GFX_VERSION=11.0.0`（借用 gfx1100 内核） | 21.8 s | 0.64 | 结果不可复现：GPU 两次之间 corr 0.988，与 CPU 之间 0.86–0.998 |
| `ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library`（用系统内核） | 118 s | 3.49 | 数值好些，但比 CPU 慢 3.5 倍 |

即便 arch override 能跑，日志里持续刷 `MIOpen ... Solver <GemmFwdRest>,
workspace required: ... provided ptr: 0 size: 0`，说明 conv 走了回退 solver。

### 2. 自研 WGSL kernel 能拿到 9× 加速

`submodule/demucs-wgpu`（上游 `eclipse005/demucs-wgpu`，fork 在
`Nahida-aa/demucs-wgpu`）是完整的 HTDemucs 前向实现，conv / attention / STFT /
overlap-add 全自研，不依赖 libtorch、ROCm、CUDA。

数值可信度：

- Vulkan vs 它自己的 CPU 后端：corr **1.0000**，SDR 103–119 dB
- Vulkan vs `demucs-tch`：drums 0.9979 / other 0.9955 / vocals 0.9989，bass 0.7492
  （bass 是弱 stem，rms 本身只有 0.0005）
- 我们封装后的输出与上游 CLI 逐样本一致（`maxdiff 0.000000`）

### 3. 结论：瓶颈在通用栈而非模型

同一份模型，换实现相差近 10 倍。参考同机反例：whisper.cpp Vulkan 转录 RTF 0.081，
远快于 CPU——「核显 + Vulkan 一定慢」在这台机器上不成立。

## 仓库内实现

| 位置 | 说明 |
|------|------|
| `submodule/demucs-wgpu` | 上游实现（fork），原始 CLI |
| `packages/demucs-wgpu` | 封装：与 `demucs-tch` 同构的 CLI、自动剥前缀的权重加载、`shifts=0` 默认 |
| `packages/demucs-tch` | tch-rs CPU 基线，支持 cpu/cuda/rocm 构建矩阵 |
| `packages/demucs-burn-tch` | Burn 0.21 + burn-tch 对比基线 |
| `submodule/demucs` | 官方 PyTorch 参考（校验权重与数值） |

## 复现方式

```bash
# 自研 Vulkan 路径（权重：data/models/demucs/htdemucs.safetensors 或 ft 目录）
just build && just run /tmp/input.wav /tmp/out htdemucs_ft   # 在 packages/demucs-wgpu

# tch 基线
cargo build --release -p demucs-tch   # 独立 workspace，见 packages/demucs-tch/justfile
```

## 未做 / 留下的问题

- `htdemucs_ft` 在 wgpu 上单位时间收益低于单模型（4 个网络只快 7.3× 而非 4× 并行），
  每 chunk 的 buffer 分配与多网络串行仍有优化空间。
- `demucs-wgpu` 尚未接入 CI 发布流水线（当前只在本地构建）。
- 未验证 NVIDIA 独显上的对比（理论上 CUDA + PyTorch 会更省事）。
