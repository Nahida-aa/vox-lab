---
number: 9
title: demucs-tch 按后端分别构建（cpu / cuda / rocm），不做运行期 --device
status: accepted
date: 2026-11-16
---

# 9. demucs-tch 按后端分别构建（cpu / cuda / rocm）

## 状态

已接受。CPU 变体已构建并复测（输出与改动前逐样本一致）；CUDA / ROCm 变体的
构建入口（justfile + CI 矩阵）已就绪，但因官方 2.15.0 预编译包尚未发布，
尚未产出二进制。

## 背景

`demucs-tch` 目前的设备是硬编码 `Device::Cpu`，而用户群里有 CUDA 与 ROCm
两类需求。与 Burn 不同，`tch` **没有** `cuda` / `rocm` feature：

- `torch-sys-0.27.0/build.rs:446` 探测 libtorch 目录里是否存在 `libtorch_cuda.so`
- `torch-sys-0.27.0/build.rs:452` 探测是否存在 `libtorch_hip.so`
- 存在则链接 `torch_cuda` / `torch_hip` + `c10_hip`

也就是说后端能力由**构建时链接的 libtorch** 决定，一次 `cargo build` 只能得到
一个后端。Cargo feature 与 `[[bin]]` 都无法在同一次构建里区分不同 libtorch，
因此「一个 package 里放 demucs-tch-cpu / -cuda / -rocm 三个 bin」做不到。

另一个约束：`torch-sys-0.27.0/build.rs:13,166` 硬校验 libtorch 版本必须是
`2.15.0`。实测用 `LIBTORCH_BYPASS_VERSION_CHECK=1` 硬配 2.14.1+cpu，**编译期
即失败**：生成的 C++ 绑定里调用了 2.15 才有的 `torch::_scaled_addmm_out`
（`libtch/torch_api_generated.cpp`），2.14.1 头文件里不存在。

也没有「降 tch 配 2.14」的退路：torch-sys 0.26.0 对应 2.13.0、0.27.0 对应
2.15.0，crates.io 上没有对应 2.14 的版本（0.27 / 0.26 / 0.24 …）。

实测 `download.pytorch.org` 的预编译包（2026-11-16）：

| 后端 | 最高可用版本 |
|------|--------------|
| CPU | 2.15.0+cpu |
| CUDA | 2.14.1+cu126 / cu130 |
| ROCm | 2.14.1+rocm7.2 |

## 决策

1. **一份源码、三次构建**：仍是单个 crate 单个 bin（`packages/demucs-tch`），
   由构建时传入的 `LIBTORCH` 决定后端，产物重命名为
   `demucs-tch-cpu` / `-cuda` / `-rocm`。
2. **不加 `--device` 参数**：设备在构建期就已确定。运行时取
   `Device::cuda_if_available()` —— CPU 变体恒为 `Cpu`，GPU 变体在有设备时
   自动落到 `Cuda(0)`，无设备时回落 `Cpu`。CLI 契约不变。
3. **构建入口**：`packages/demucs-tch/justfile` 提供
   `build-cpu` / `build-cuda` / `build-rocm`，各自使用独立 `--target-dir`
   （`target/cpu`、`target/cuda`、`target/rocm`），产物落在 `dist/`。
4. **CI**：`release-demucs-tch-linux.yml` 与 `-windows.yml` 改为 backend 矩阵。
   CUDA / ROCm job 先探测 2.15.0 预编译 libtorch 的 URL，404 则跳过（不标红），
   官方发布后自动生效，不需要改 workflow。
5. **资产形态**：CPU zip 继续平铺 libtorch `.so` / `.dll`；CUDA / ROCm zip 只含
   可执行文件，由使用者自备同版本 libtorch（CUDA/ROCm 版 libtorch 是 GB 级）。

## 后果

- 权重加载改成在目标设备上物化：`TensorStore` 携带 `device`，`st_to_tensor`
  末尾 `.to_device()`；`create_sin_embed` / `create_2d_sin_embed` 的结果显式搬到
  输入设备；`tensor_to_vec` 回读前列 `.to_device(Device::Cpu)`。
  这些改动在 CPU 下不改变数值（与改动前输出 corr 1.0000）。
- 每加一个后端就多一份完整的 libtorch 构建缓存（`target/<backend>`），
  磁盘占用增加；不要删掉 `--target-dir` 否则每次都会重编 `torch-sys`。
- 若将来官方发布 2.15.0 的 CUDA/ROCm libtorch，只需在矩阵 URL 里改版本号。
  在那之前 CUDA/ROCm 变体只有三条路：等官方预编译包；自己从源码编 libtorch
  2.15 的 CUDA/ROCm 版本（数小时 + 数十 GB）；fork torch-sys 针对 2.14 重新
  生成绑定（脆弱，不建议）。**不要**尝试「降 tch 配 2.14」—— 没有对应版本。
- `demucs-burn-tch` 不受影响：它走 Burn 的 `LibTorch` 后端，
  `LibTorchDevice::default()` 仍是 `Cpu`，只用于性能对比。

## 相关

- [0002](./0002-no-scheduler-compile-time-device.md) Burn 侧 Device 由 feature 决定（不同机制）
- [0008](./0008-demucs-tch-standalone-workspace.md) demucs-tch 独立 workspace
- `packages/demucs-tch/justfile`、`.github/workflows/release-demucs-tch-*.yml`
