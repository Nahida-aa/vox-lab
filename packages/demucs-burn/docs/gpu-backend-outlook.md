# GPU 路径与上游变化（2026-11-16 核对）

本文记录一次面向「Demucs 能否稳定跑 GPU」的探索结论：本机现状、Burn 升级预期、
`burn-tch` 的去留，以及从 AMD 侧还能做什么。基准数字仍以
[`packages/research/separate/readme.md`](../../../research/separate/readme.md)
为准，本文不重复堆数据。

## 结论

1. **已验证可用的 GPU 路径只有 Vulkan/wgpu**；ROCm 10.1 本身能识别并驱动
   `gfx1103`，但 Burn 的 rocm 后端因 `cubecl-hip-sys` 版本上限编译失败。
2. **`demucs-burn-tch` 永远是 CPU**：binary 传 `Default::default()`，而
   `LibTorchDevice::default()` 就是 `Cpu`（见 `burn-tch-0.20.1/src/backend.rs`）。
3. **Burn 0.21/0.22 的更新方向与「wgpu 慢」的归因一致**，升级后值得重测；
   但 GPU hang 的根因在 amdgpu MES，Burn 只能降低触发概率，不能保证修复。
4. **`burn-tch` 已进入删除流程**：0.22 弃用，main 已移除，0.23 起不再提供。
   生产 CPU 路径需要提前规划（钉版本 / 直接用 `tch-rs` / 换 `burn-flex`）。

## 本机环境（2026-11-16）

| 项 | 值 |
|---|---|
| 发行版 | Garuda Linux（Arch 系，rolling） |
| 内核 | 7.2.15-zen1-1-zen |
| Mesa / RADV | 26.2.7 |
| linux-firmware | 20261019 |
| ROCm | 10.1（HIP 7.16；rocBLAS 10.1-2；`rocminfo` 可见 `gfx1103`） |
| GPU | Radeon 780M（RADV PHOENIX）+ llvmpipe 软件适配器 |
| amdgpu 关键参数 | `uni_mes=1`、`gpu_recovery=-1`（auto）、`lockup_timeout` 空（默认 2000ms） |

判断：内核 / Mesa / 固件已经比社区报告里「MES 修复落地」的版本更新
（修复窗口大致是 6.11–6.13，另有 6.18.9 修复 780M pageflip 的报告），
所以**继续靠「升级 AMD 栈」解决 hang 的空间已经不大**。

## 如何确认真的走了 GPU

```bash
# 1. 编译 wgpu 后端
cargo build --release -p demucs-burn \
  --bin demucs-burn-wgpu --no-default-features --features wgpu

# 2. 强制 780M，避免选中 llvmpipe；warmup 预编译 shader
CUBECL_WGPU_DEFAULT_DEVICE='IntegratedGpu(0)' \
AMD_VULKAN_ICD=RADV \
./target/release/demucs-burn-wgpu \
  --warmup --tasks-max 128 --model htdemucs_ft input.wav out/

# 3. 校验：设备名与 GPU 占用
vulkaninfo --summary | grep -i deviceName
watch -n1 'cat /sys/class/drm/card*/device/gpu_busy_percent'
```

- `--warmup` 只对 wgpu/vulkan 生效（代码里 `cfg(any(feature = "wgpu", feature = "vulkan"))`）。
- `--tasks-max` 只影响 wgpu 的 CPU command recording，对 tch 无效。
- 不要开 `autotune`：在 RADV + 780M 上会触发驱动 hang（见 `cpu-fallback.md`）。

## Burn 升级预期

| 版本 | 与 Demucs 相关的改进 | 对本仓库的含义 |
|---|---|---|
| 0.21.0（2026-05） | Vulkan/wgpu eager launch overhead 平均降 3.4×，fusion 平均 5.4×，最高 8.2×；重做 device handle | 直接对症「wgpu 慢 / 小 kernel 多」 |
| 0.22.0（2026-10） | transposed conv 直接 kernel、strided pointwise 走 matmul、自适应内存池、WGPU 软件 graph replay、roofline autotune | 对症 Demucs 的 conv/conv_transpose |
| 0.22.1（2026-10-20） | CubeCL 0.11.1 / CubeK 0.3.1，fusion 修复 | 当前最新稳定版 |
| 0.23.0-pre.2（2026-11-09） | 预发布 | 移除 libtorch 后端后的版本，不建议现在跟 |

升级成本：0.22 移除了 backend 泛型（`Model<B>` → 运行时 `Device` 选择），
是一次较大的 API 迁移；0.21 相对温和。`submodule/demucs-rs` 的 `origin/main`
已有 `182afa0 chore: upgrade to burn 0.21`，本仓库指针仍停在 Burn 0.20.1。
另外 `submodule/voxcpm-rs/vendor/patched/burn-cubecl` 有对 0.20.1 conv
累加器的补丁，升级时需要一起复核。

## `burn-tch` 的时间线

| 时间 | 事件 |
|---|---|
| 2026-08-31 | PR #5525：`burn-tch` 标记 deprecated。PR 明确「构建/测试/发布不变，弃用窗口内继续支持」 |
| 2026-10-06 | 0.22 发布说明：Candle 已移除，NdArray 与 LibTorch 进入弃用 |
| 2026-10-08 | PR #6009：`chore!: remove deprecated libtorch backend` 合并进 main |
| 0.23 | 将不再包含 `burn-tch` / `LibTorch` 后端 |

原因（官方表述 + 可核对事实）：

1. 后端向 CubeCL 收敛（cuda/rocm/wgpu/cpu），`burn-flex` 成为纯 Rust CPU 参考实现；
2. 贡献者指南不再要求新算子实现 tch 版本；
3. tch 依赖外部 libtorch，维护成本高、算子覆盖容易掉队（如 #5924 的 `prelu` 维度 bug）；
4. 同一轮整合里 Candle 后端先被删除（#5343），libtorch 是下一个。

对本仓库的三个选项：

- **钉 `burn 0.22.1`**：短期最省事，仍可用 deprecated 的 tch；
- **直接用 `tch-rs`**：脱离 Burn 保留 libtorch + MKL 的 CPU 快路径；
- **迁 `burn-flex` 或 CubeCL CPU**：flex 是 eager、无 fusion/autotune，
  conv 重的 Demucs 大概率不如 MKL，需实测。

## AMD 侧还能做什么

### 1. Vulkan 仍是当前默认选择，但 ROCm 并非「缺 kernel」

**已核对本机，修正一条社区结论**：社区有说法称 ROCm 7.2.x 不含 gfx1103 的
TensileLibrary，HIP 只能跑 fallback。本机在 7.2.4 与升级后的 10.1 上都不成立：

```bash
$ ls /opt/rocm/lib/rocblas/library/ | grep -c gfx1103
150
$ ls /opt/rocm/lib/rocblas/library/ | grep -i "TensileLibrary.*gfx1103" | head -2
TensileLibrary_lazy_gfx1103.dat
TensileLibrary_Type_4xi8I_HPA_Contraction_..._gfx1103.hsaco
```

即 Arch 打包的 `rocblas 10.1-2` 已包含 gfx1103 的 Tensile 资产，
`rocminfo` 也能看到完整 `gfx1103` agent。最小冒烟（ROCm 10.1）：

```bash
$ rocblas-bench -f gemm -r f32_r -m 2048 -n 2048 -k 2048 --iters 20
Device ID 0 : AMD Radeon 780M Graphics gfx1103
N,N,2048,2048,2048,..., 465.705 GFlops, 36890 us
```

`rocm-smi` 也能看到设备（APU 上会有一条 `Exception caught: map::at` 警告，
不影响设备枚举）。所以「ROCm 跑不了 780M」不是当前事实，只是**尚未在本仓库的
Demucs/Burn 路径上验证过**。

当前仍把 Vulkan 作为默认，理由是可复现的实测：whisper.cpp Vulkan RTF 0.081，
而 Demucs wgpu RTF 2.41 已经能跑；ROCm 路径在本仓库还没有成功基准。

### 2. ROCm 版本：系统更新后已到 10.1

- 2026-11-16 系统更新后，`rocm-core` / `hip-runtime-amd` / `hsa-rocr` /
  `rocm-hip-sdk` 均为 `10.1`，HIP 7.16，`rocblas 10.1-2`；`rocminfo`、`rocm-smi`
  与 `rocblas-bench` 都能识别并使用 `gfx1103`。
- 因此**不需要再为「7.2.4 太旧」而升级**：10.1 已比之前的 7.14 讨论版本更新，
  也自带 gfx1103 的 Tensile 资产。
- Burn 的 `rocm` 后端走 CubeCL HIP，还需验证的是 **CubeCL/HIP 与本机 ROCm 10.1
  的兼容性**；本仓库此前在 780M 上记录到 GPU Hang，需在 10.1 上重新验证。
- 不建议用 `HSA_OVERRIDE_GFX_VERSION` 硬覆盖：`rocminfo` 已识别 `gfx1103`，
  覆盖只会掩盖真实问题。

**但 Burn 的 `rocm` 后端目前编译不过（2026-11-16 实测）**：

```bash
$ cargo build --release -p demucs-burn --bin demucs-burn-rocm \
    --no-default-features --features rocm
error[E0425]: cannot find function `hipSetDevice` in crate `cubecl_hip_sys`
... 55 previous errors
```

原因不是本仓库代码，而是 `cubecl-hip-sys 7.14.6085001` 的 bindings 最高只到
`hip_60850`；ROCm 10.1 的 `hipconfig --version` 返回 `7.16.0-0000000`，
build.rs 按补丁号选择的 feature 不存在，导致整个 `cubecl_hip_sys` 没有任何符号。
cubecl-hip-sys 当前最新版本仍是 `7.14.6085001`，Burn 0.22 也依赖同一版本，
**所以「升级 Burn」短期内也绕不过去**。可行路径只剩：给 cubecl-hip-sys 打
vendored patch 强制 `hip_60850`（ABI 风险），或改用 PyTorch ROCm 跑 Demucs。

**上游原因（不是故意不支持）**：cubecl-hip 的 issue #39 已说明，crate 版本和
bindings 都按 HIP **patch 号**命名（`7.14.6085001` = HIP 7.14 / patch `60850`）。
从 HIP 7.15（ROCm 10.x）起，ROCm 把 `VERSION` 的 patch 置 0，构建时替换成
提交日期戳（`%y%U%w`，如 `26333`），旧的 patch 命名机制无法覆盖任何新版本。
issue #39 提议改成按 `major.minor` 命名，尚未合并；cubecl #922 的 HIP 构建失败
issue 也仍然打开。截至 2026-11-16，cubecl-hip 仓库**没有任何相关 PR**：
open PR 列表为空，也没有 `hip_715/hip_716` 分支；最后一个 bindings PR 是
#36（`hip_60850` / ROCm 7.14，2026-07-30 合并）。历史上每个 ROCm 版本都有补
bindings 的 PR（#29–#36），所以这是**版本方案变更 + 维护滞后**，不是刻意放弃 ROCm。

### 3. 如果 hang 仍复现：收集证据，而不是继续盲调

先确认是不是真的 MES 挂死，再谈规避：

```bash
# MES 日志（下一次复现前打开）
# /etc/modprobe.d/amdgpu.conf: options amdgpu mes_log_enable=1
sudo journalctl -k -b | grep -iE "amdgpu.*(mes|timeout|reset|hang)"
sudo cat /sys/kernel/debug/dri/0/amdgpu_firmware_info

# 让驱动尽量 recover 而不是整机卡死（默认 auto 已是开启方向）
# 临时实验：amdgpu.gpu_recovery=1
# 长 kernel 误判可临时放宽：amdgpu.lockup_timeout=-1 或 20000
```

可做的隔离实验（一次只改一个变量）：

- `amdgpu.uni_mes=0`：关闭 Unified MES，验证是否与 unified scheduler 相关
  （本机默认 `uni_mes=1`；这是实验项，不是已知修复）。
- `amdgpu.lockup_timeout=20000`：排除长 kernel 被 2s 超时误杀。
- 换 Mesa 版本 / 换内核（zen ↔ 主线）复现：确认是否 RADV 特定路径。
- 若只在 Burn autotune 打开时挂：固定为「禁用 autotune」，并把复现日志
  报到 Mesa / amdgpu，而不是继续在应用层绕。

### 4. 不建议的方向

- 用 `amdgpu.dcdebugmask` 之类显示相关参数治 compute hang：不对症。
- 为了 GPU 而把整个 Demucs 迁到 ROCm：收益尚未实测，且 Burn 的 ROCm 后端
  还多一层 CubeCL 版本约束；应先小规模 A/B 再决定。
- 把 `burn-tch` 当作长期 GPU 方案：它默认 CPU，且 0.23 起被移除。

## 重测清单

1. 升级 Burn 到 0.22.1（或先只升 0.21），重跑 `packages/benchmark/separate/`。
2. 固定 `htdemucs_ft` + 同一段音频，对比：tch CPU、wgpu（warmup、
   `tasks_max=128`、autotune off）、可选 ROCm（当前被 cubecl-hip-sys 阻塞，
   见上节）。
3. 记录 RTF p50/p95、峰值内存、是否 hang；hang 时保留 `journalctl -k` 全段。
4. 只有 wgpu RTF 稳定低于 tch CPU，才考虑把 GPU 设为默认路径。

## 参考

- [Burn 0.21.0：launch overhead 3.4×/5.4×/8.2×](https://burn.dev/blog/release-0.21.0/)
- [Burn 0.22.0：conv 优化、内存池、graph replay](https://tracel.ai/blog/release-0.22.0/)
- [PR #5525：deprecate burn-tch](https://github.com/tracel-ai/burn/pull/5525)
- [PR #6009：remove deprecated libtorch backend](https://github.com/tracel-ai/burn/pull/6009)
- [ROCm 7.14 安装：Ryzen APU（含 gfx1103）要求 OEM kernel 6.14](https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html?fam=ryzen&gfx=gfx1150&i=tar&os=ubuntu&w=compute)
- [amd-torch-device-gfx1103 wheels](https://rocm.nightlies.amd.com/v4/whl/amd-torch-device-gfx1103/)
- [ROCm 7.2.x 无 gfx1103 Tensile（社区说法，本机实测不成立）](https://nunix.dev/tech/local-ai-rocm-journey/)
- [amdgpu 模块参数（uni_mes / gpu_recovery / lockup_timeout）](https://www.kernel.org/doc/html/v7.1-rc4/gpu/amdgpu/module-parameters.html)
- [780M MES timeout 与内核修复窗口（社区记录）](https://ataary.com/ubuntu-linux-with-ollama-rocm-on-amd-ryzen-780m-igpu/)
