---
number: 8
title: demucs-tch 独立 workspace（避开 torch-sys links 冲突）
status: accepted
date: 2026-11-16
---

# 8. demucs-tch 独立 workspace（避开 `torch-sys` links 冲突）

## 状态

已接受。Stage 0 已验证：独立 workspace 下 `tch 0.27` / `torch-sys 0.27` /
libtorch 2.15 下载、编译、链接成功，`--help` 可启动，`Cargo.lock` 仅一个
`torch-sys 0.27.0` 且 **0 个 burn 包**。

## 背景

新建 `packages/demucs-tch`，用 `tch-rs` 直接实现 Demucs 前向，脱离 Burn
（见 [0002](./0002-no-scheduler-compile-time-device.md) 的 Rust 库形态、
`burn-tch` 被 deprecated 的背景）。目标是 pin 较新的 `tch 0.27`
（crates.io，2026-11-10，对应 PyTorch 2.15），启用 `download-libtorch`。

主 workspace 里 `demucs-burn` / `voxcpm-burn` / `submodule/voxcpm-rs` 依赖
`burn 0.20.1`，而 `burn` 的传递依赖 `burn-tch 0.20.1` → `tch 0.22` →
`torch-sys 0.22`。`torch-sys` 声明 `links = "tch"`，**同一依赖图里只能有一个
版本**，因此 `tch 0.27`（`torch-sys 0.27`）与 `tch 0.22`（`torch-sys 0.22`）
冲突，cargo 在版本选择阶段直接报错：

```
package `torch-sys` links to the native library `tch`, but it conflicts with a
previous package which links to `tch` as well
```

## 实测过的方案（均失败，记录以免重蹈）

| 尝试 | 结果 |
|------|------|
| 删除 `demucs-burn-tch` / `voxcpm-burn-tch` 两个 bin | `burn 0.20.1` 仍拉 `burn-tch` |
| 把 `voxcpm-burn` / `voxcpm-rs` / 两个 patched crate 移出主 workspace | `demucs-burn` 0.20.1 仍在 → 仍冲突 |
| `demucs-burn` / `demucs-core` 升 Burn 0.22.1 | **`burn-dispatch 0.22.1` 硬依赖 `burn-tch ^0.22.1`（非 optional）** → 仍拉 `tch 0.22` |
| 删 `demucs-core` 的 `tch = ["burn/tch"]` feature | `burn-dispatch` 无条件拉 `burn-tch`，无效 |
| `resolver = "2"` | 无效 |
| 删 `Cargo.lock` 重新生成 | 无效（不是陈旧 lock） |

关键事实：`burn-tch` 在 0.20 和 0.22 都标 deprecated 但**仍被构建/发布**，
且 `burn-dispatch 0.22.1` 把它作为硬依赖 —— 所以「升级 Burn」解不了冲突。

## 决策

- `packages/demucs-tch/Cargo.toml` 带 `[workspace]` 空表，**自成独立 workspace**。
- 根 `Cargo.toml` 用 `workspace.exclude` 显式列出 `packages/demucs-tch`
  （以及按用户要求移出的 `packages/voxcpm-burn`），不列入 `members`。
- 两套 `torch-sys` 不进同一依赖图，`demucs-tch` 用 `tch 0.27` / libtorch 2.15，
  主 workspace 保持 Burn 0.20.1 与 `voxcpm-rs` 的 bf16 vendored patch 原样不动。
- 由此 `demucs-tch` 不共享主 workspace 的 `[workspace.dependencies]`、
  `target/` 和 `Cargo.lock`：依赖直接写版本，产物在
  `packages/demucs-tch/target/`，CI 用 `working-directory` 或 `--manifest-path` 构建。

## 后果

- 新 crate 的 `Cargo.lock` 独立（应入库），依赖版本不与主 workspace 统一，
  升级 `tch-rs` 时只影响它自己。
- 主 workspace 的 `voxcpm-rs` bf16 / RDNA4 修复（vendored `burn-cubecl 0.20.1`）
  完全不受影响 —— 这是选择独立而非升级 Burn 的主要原因。
- 若将来主 workspace 彻底移除 `burn-tch`（例如全部迁移到 Burn 0.23+ 且
  `burn-dispatch` 不再硬依赖它），可重新评估合并回 `members`，届时写新 ADR。
- 不要仅因为「少一个 workspace」就把 `demucs-tch` 加回 `members` —— 会立刻
  复现 `links = "tch"` 冲突。

## 数值验证（PyTorch 参考，2026-11-16）

`packages/demucs_pytorch/` 是专门校验 `demucs-tch` 的 PyTorch 参考工具
（core=官方模型加载/推理，cli=对比命令，server=可选 HTTP）。用官方
`demucs.pretrained.get_model('htdemucs')` + torch hub `.th` 权重对同一
`packages/benchmark/ref/short.wav` 产出参考 stems，与 `demucs-tch` 输出逐样本比对：

| stem | 相关系数 | SDR (dB) | 备注 |
|------|---------|----------|------|
| drums | 0.56 | -5.65 | rms ~0.002，信号极弱 |
| bass | 0.75 | 1.90 | rms ~0.001 |
| other | **0.9986** | **25.29** | 占信号能量 ~99%，主干前向对齐 |
| vocals | 0.46 | -1.75 | rms ~0.001，信号极弱 |

**结论**：主导能量所在的 `other` 与官方 PyTorch 高度吻合，说明模型前向
（conv / transformer / 权重 key map / STFT+CaC / freq+time 相加）整体正确；
short.wav 为合成噪声，官方实现同样把绝大部分能量归到 `other`。
弱能量 stem 相关系数偏低，主要来自实现差异：官方 `BagOfModels` 的
`shifts` / overlap-add 后处理与 `demucs-tch` 单次 segment + triangular window
不同，且小幅度信号对数值噪声极敏感。**不宜据此宣称逐 stem 完全对齐**，
若需更强结论应改用人声/音乐等真实音频并统一后处理再比对。

## 相关

- [0002](./0002-no-scheduler-compile-time-device.md) Device 由编译期 feature 决定
- [0003](./0003-rust-cli-python-http.md) Rust 侧库 + CLI 形态
- `packages/research/separate/readme.md` 的「上游更新」记录了 Burn 0.21/0.22 的性能改进
- `packages/demucs_pytorch/` 的 `cli.py` 提供上述对比命令
