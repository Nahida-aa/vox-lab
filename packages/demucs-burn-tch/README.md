# demucs-burn-tch (性能对比基线)

`demucs-burn` 的 `LibTorch` CPU 后端二进制，**仅作为 `demucs-tch` 的性能对比基线**长期保留。
它复用 `packages/demucs-burn` 的 `Cli` / `run` 逻辑（path 依赖，加一个薄壳 bin），
跑在 Burn 0.21 的 `burn-tch` 后端上。

## 为什么是独立 workspace

`burn-tch 0.21` 依赖 `tch 0.22 / torch-sys 0.22`，而 `packages/demucs-tch` 用
`tch 0.27 / torch-sys 0.27`。两者都声明 `links = "tch"`，同一依赖图里只能有一个，
因此本包带 `[workspace]` 空表、不列入根 `Cargo.toml` 的 `members`（列入 `exclude`）。
详见 `docs/decisions/0008-demucs-tch-standalone-workspace.md`。

`demucs-burn` 作为 path 依赖使用 `clap.workspace = true` / `anyhow.workspace = true`，
所以本包在自己的 `[workspace.dependencies]` 里镜像了这两项。

## 构建（release，用于性能对比）

`torch-sys` 的 `download-libtorch` 在本机常因网络失败，指向已下载的 libtorch 2.9：

```bash
LIBTORCH=<repo>/target/debug/build/torch-sys-*/out/libtorch/libtorch \
  cargo build --release --bin demucs-burn-tch
```

运行时同样要 `LD_LIBRARY_PATH=<同上>/libtorch/lib`。

## 跑性能对比

```bash
LD_LIBRARY_PATH=<libtorch-2.9>/lib ./target/release/demucs-burn-tch \
  --model htdemucs --model-dir /path/to/models/demucs \
  --benchmark-rounds 2 input.wav out_dir/
```

读取 stdout 的 `Benchmark-Gen-Time-Round{N}`。与 `demucs-tch` 用同一输入、
同一模型/权重、各自 release、串行（不要并行，会抢 CPU）对比。

## 实测结果（2026-11-16，short.wav 10s / 44.1kHz，串行独占 CPU）

| | demucs-burn-tch (Burn 0.21 / libtorch 2.9) | demucs-tch (tch 0.27 / libtorch 2.15) |
|---|---|---|
| Load | 0.516s | 0.323s |
| Gen (2 轮) | 12.06s / 10.94s | 12.43s / 11.48s |
| RTF | ~1.1–1.2 | ~1.15–1.24 |

两者基本持平（差 ~3–5%，噪声范围）。**用 tch-rs 脱离 Burn 没有可测量的性能代价**。
注意这是「两条不同 Rust 实现 + 不同 libtorch 版本」的端到端对比，不是纯 backend 对比。
