# Audio Separation

## Backend comparison

All benchmarks on `htdemucs_ft`, `tasks_max=1`. CPU: Ryzen 7 7840HS. GPU: Radeon 780M (RADV). RTF computed from generation time only (excluding load & warmup).

| Backend | Device | Audio | RTF (gen only) |
|---------|--------|-------|:--------------:|
| Burn wgpu | wgpu (RADV) | short (10s) | **2.77** |
| Burn wgpu | wgpu (RADV) | medium (60s) | **2.41** |
| Burn tch | libtorch CPU | short (10s) | **1.66** |
| Burn tch | libtorch CPU | medium (60s) | **1.26** |
| ONNX (onnxruntime-node) | Node CPU | any (4-stems) | ~6.2 | — |
| ONNX (onnxruntime-node) | Node CPU | any (vocals) | ~2.2 | — |

> **结论适用范围（重要）**：以上数字是在 **Burn 0.20.1 / CubeCL 0.9.0 + Radeon 780M
> (RADV, gfx1103)** 这一具体组合下测得，反映的是**当时 wgpu 后端的成熟度**，
> **不是 Demucs 模型「天生不适合 GPU」**。理由见下方「归因：后端而非模型」。
> 硬件/驱动也有限定：780M 上整网 GPU 推理会 MES hang，wgpu 是靠禁用 CubeCL
> autotune 才跑起来的。换用独显、更新驱动或升级 Burn，结论可能反转。
> 同机反例：whisper.cpp Vulkan 转录 RTF **0.081**，远快于 CPU —— 说明
> 「核显 + Vulkan 一定慢」在这台机器上并不成立，差别在后端实现。

## 归因：后端而非模型

**结论**：`burn wgpu` 慢于 `burn tch`，主因是 **Burn 0.20.1 / CubeCL wgpu 后端的
conv kernel 与 GPU 调度开销不成熟**，不是 Demucs 的算子尺寸（「小卷积」）问题。

证据：

1. **同一份算子代码，只换 backend，RTF 差近 2 倍**。`submodule/demucs-rs/demucs-core/src/model/conv.rs`
   全部使用 `burn::nn::conv::{Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d}`，
   纯泛型 `B: Backend`，无自写 kernel。kernel 为 `[8,1]` / `[3,3]` / 8，通道 48→96→192→384，
   属常规中等规模卷积，并非「小到 launch overhead 占主导」的逐点卷积。
   tch 走 libtorch/MKL（1.26），wgpu 走 CubeCL/Vulkan（2.41）——模型未变，
   故差距来自 backend 实现而非模型特征。
2. **框架侧已知缺陷**。`submodule/voxcpm-rs/vendor/patched/burn-cubecl` 里就有针对
   `burn-cubecl 0.20.1` direct conv / conv_transpose2d **累加器**的补丁，说明该版本
   conv kernel 本身存在已需修补的问题（补丁针对 bf16/RDNA4，本仓库 wgpu 走 f32，
   且根 `Cargo.toml` 未启用 `[patch.crates-io]`，用的是 crates.io 原版）。
   另有记录：`cpu`(CubeCL MLIR) 后端 per-kernel JIT 过慢、autotune 会触发 RADV
   驱动 hang 必须禁用、fusion 只覆盖 elementwise 因而对 conv 无效。
3. **CPU 侧 command recording 也是瓶颈**：`--tasks-max` 1→128 带来 +23%（34.4s→26.3s）。

**「fusion 无效 → 瓶颈在 conv」只是推断，不是实测**：fusion 无效只能证明瓶颈不在
elementwise，无法排除 CrossTransformer attention（`cpu-fallback.md` 指出它是 GPU hang
主因之一）、Wiener 滤波 / STFT 等 DSP、或 CubeCL 的 kernel launch 调度。
**尚未做 kernel 级 profiling**，因此无法给出各因素占比。

> **待验证**：升级 Burn 后重跑基准，见下方「上游更新」。

## Burn wgpu

`vox-lab/packages/demucs-burn/` — Thin Rust binary wrapping demucs-core with Burn backend.

### Install (demucs-burn 已迁至 vox-lab, 经 GitHub Release 下载)

```bash
# 已发布后端: tch / wgpu (下载 + 校验 sha256, 平铺到 data/bin, 见 env ensure)
cargo run -p cli -- env ensure demucs_burn_tch_bin   # tch (libtorch CPU/MKL, 含 libtorch.so 平铺)
cargo run -p cli -- env ensure demucs_burn_wgpu_bin  # wgpu (GPU via Vulkan/Metal/DX12)

# 未发布后端 (cpu/cuda/vulkan/rocm) 无 release 资产, 不支持
```

源码仓库: `vox-lab/packages/demucs-burn`（依赖 `vox-lab/submodule/demucs-rs`）。如需本地编译:

```bash
# wgpu (GPU via Vulkan/Metal/DX12, default) — works
cargo build --release -p demucs-burn --bin demucs-burn-wgpu --no-default-features --features wgpu

# tch (libtorch CPU via MKL — best CPU path, requires LD_LIBRARY_PATH)
LIBTORCH_DIR=$(find target/release/build/torch-sys-*/out/libtorch/libtorch/lib -maxdepth 0)
LIBTORCH_DIR=$LIBTORCH_DIR cargo build --release -p demucs-burn --bin demucs-burn-tch --no-default-features --features tch
```

| Binary | Size | RTF (medium, 60s) | RTF +fusion | Notes |
|--------|:----:|:------------------:|:-----------:|-------|
| `demucs-burn-wgpu` | 37MB | **2.41** | **2.40** | GPU via Vulkan, +16s warmup first run |
| `demucs-burn-tch` | 4.0MB | **1.26** | — | CPU via MKL, no warmup overhead, needs `LD_LIBRARY_PATH` |
| `demucs-burn-cpu` | 177MB | — | — | CubeCL MLIR, per-kernel JIT too slow for demucs |
| `demucs-burn-rocm` | 32MB | — | — | `GPU Hang` on 780M MES, needs gfx9+ ROCm |

### Runtime

- model load: ~1.0s (tch), ~1.2s~1.5s (wgpu)
- warmup (`--warmup`): ~16s wgpu, pre-compiles CubeCL shaders for all 4 htdemucs_ft models. Reduces first-inference RTF by ~15%.
  Without warmup, shader compilation happens lazily during inference, increasing first-run RTF (short 3.44, medium 2.68).
  With warmup, inference-only RTF is **2.63** (short) / **2.41** (medium, with `tasks_max=128`).
- 多轮 benchmark (`--benchmark-rounds N`): 不重启进程跑 N 次推理，验证 CubeCL JIT 是否存在冷启动之外的残余开销。
  实测在 warmup 后第二轮与第一轮生成时间差 <0.3%（26.34s vs 26.58s，10s 音频）。
  **结论：warmup 已全覆盖所有 shader，无残余 JIT。单轮 benchmark 即可反映稳定态性能。**
- GPU fusion (`--features wgpu,fusion`): 实测无改善（RTF 2.40 vs 2.41，在误差范围内）。fusion 对 demucs 的小 kernel 卷积和 transposed conv 无效。已确认在 RADV 780M 上不 crash。
- CUDA GPU: tch can use CUDA if available (not tested here)
- Real CPU production path: **tch** (RTF 1.26-1.66) or **ggml** (RTF 1.44-2.91)
- Default model: `htdemucs_ft` (4 per-stem specialist models, 333MB). Change with `--model htdemucs`.

### Known issues

- **CubeCL autotune** (wgpu 已禁用): causes `elemwise_fuse` pipeline failure → GPU driver hang on RADV. Fix: remove `AutotuneConfig`.
- **Burn fusion** (`--features wgpu,fusion`): 在 RADV+780M 上不 crash，但对 demucs 无性能提升（RTF 2.41→2.40，误差范围）。Burn 0.20 的 fusion 主要优化 elementwise 操作链（add+norm+act 等），覆盖不到本模型的耗时部分。⚠️ 由此只能推出「瓶颈不在 elementwise」，**不能据此断定瓶颈就在 conv/conv_transpose**；未做 kernel 级 profiling，见「归因：后端而非模型」。
- **wgpu ≠ Vulkan**: wgpu is cross-platform; on Linux this uses Vulkan via RADV.
- **GPU memory on 780M**: 5.86 GiB device-local + 11.73 GiB host-visible. 120s wgpu stable.
- **tch libtorch**: needs `LD_LIBRARY_PATH` pointing to libtorch lib dir at runtime.
- **cpu**: impractical for demucs in **Burn 0.20.1** — CubeCL MLIR JIT compiles each kernel individually, conv/matmul unoptimized. 属当时后端限制，0.21+ 已大幅改进，见「上游更新」。
- **rocm**: `GPU Hang` on 780M (MES firmware issue), requires gfx9+ with stable ROCm.

### tasks_max tuning (wgpu only)

`--tasks-max` controls CPU threads for wgpu command recording. **不影响 tch/libtorch 后端。**

| tasks_max | short (10s) | 提升 |
|:---------:|:-----------:|:----:|
|    1      |   34.4s     |  —   |
|   128     |   **26.3s** | +23% |

默认值已改为 128。之前的 crash 是 autotune 导致，非 tasks_max 本身问题。

## ONNX (onnxruntime-node)

`packages/cli/src/ml/demucs.ts` — Node.js wrapper using onnxruntime-node. Slow due to CPU-only ORT inference. Used as fallback when Burn wgpu is unavailable.

## 上游更新（2026-11-16 核对）

本仓库 `Cargo.lock` 锁在 **Burn 0.20.1 / burn-cubecl 0.20.1 / cubecl 0.9.0**。
上游已有多个版本直接针对本节的归因，建议升级后重跑基准：

| 版本 | 与本节相关的改进 |
|------|------------------|
| **Burn 0.21.0** (2026-05) | 官方称「4 个月改进让框架全面更快，**小模型推理上框架开销下降尤其明显**」；**重做 device handle 降低框架开销**；kernel selection 因更好的 autotune + 新校验层更可靠；新增 `burn.toml` 可不重编译调内部参数；`opt(burn-cubecl): Optimized tensors by default`。官方 blog 给出的 launch overhead 降幅：eager 平均 **3.4x**、fusion **5.4x**、最高 **8.2x**，GEMV 提升至与 LibTorch 相当或超过 |
| **Burn 0.22.0** (2026-10-06) | 大量 conv / fusion / autotune 性能工作：`perf(cubecl): NHWC direct kernel for conv_transpose2d` (#5760)、`perf(conv): route strided pointwise convolutions through the matmul path` (#5429)、`refactor(cubecl): call cubek's direct convolution routine` (#5646)、`fix(cubecl): use real out_channels in conv_transpose2d autotune key` (#5289)、autotune roofline policy、`Feat/profiling` (#5726)、graph capture。CubeCL bump 到 `48301b5` |
| **Burn 0.22.1** (2026-10-20) | CubeCL 0.11.1 / CubeK 0.3.1，多处 fusion 修复。**当前最新稳定版** |
| **Burn 0.23.0-pre.2** (2026-11-09) | 预发布，未评估 |

**子模块状态**：`submodule/demucs-rs` 的 `origin/main` 已有
`182afa0 chore: upgrade to burn 0.21`（2026-08-15），而本仓库子模块指针停在
`b54aecd`（Burn 0.20.1）。该提交说明明确列出性能收益即上述官方 blog 的
launch overhead 数据，并已把 `demucs-core` 升到 `burn 0.21` + `cubecl 0.10`、
适配 `PaddingConfig::Explicit` 非对称 API（1d 1→2 参数、2d 2→4 参数）、
`GlobalConfig` → `CubeClRuntimeConfig`。**即上游已升级，本仓库尚未跟进。**

**关于 `burn-tch` 被 deprecated（不构成升级障碍）**：Burn 0.22.0 把
`burn-tch` / `burn-ndarray` 标为 deprecated（`chore: deprecate burn-tch backend` #5525,
2026-08-31 合并）。查 PR 原文：这只是打 `#![deprecated]` 标记 + 迁移指引，
「**Nothing changes about what is built, tested or published — the crate stays
supported for the deprecation window**」，弃用窗口内照常构建/测试/发布。
迁移建议是改用 CubeCL 后端（cuda/rocm/wgpu/cpu）或 `burn-flex` 纯 Rust CPU，
原因是**维护方向**（flex 成为参考实现），**不是 tch 性能不如 Burn**。

> 本仓库生产默认的 `demucs-burn-tch` 目前通过 `burn/tch` feature 间接使用
> libtorch。即使将来不再走 `burn-tch`，也可**直接依赖 `tch-rs`** 脱离 Burn
> 使用该 CPU 路径 —— 因此 `burn-tch` 的弃用**不是升级障碍**。

**仍需处理的破坏性变更**：0.21 的 `PaddingConfig::Explicit` 非对称
（1d 1→2 参数、2d 2→4 参数）、`Shape` / `BackendTypes` 等接口变化；
`demucs-core/src/model/conv.rs` 已由上游 `182afa0` 适配，跟进子模块即可。

**结论**：本节「wgpu 慢于 tch」的归因与上游 0.21 的主攻方向（框架开销）一致，
很可能随升级改善。**升级并重跑基准前，不要把这些 RTF 当作长期结论。**

> 核对方式：上游发布与 PR 信息经 `gh api` 查询（GitHub 直连 git/curl 在本机易超时，
> 优先用已登录的 `gh cli`）。`demucs-rs` 上游 `origin/main` 最新即 `182afa0`，
> 之后无新提交。
