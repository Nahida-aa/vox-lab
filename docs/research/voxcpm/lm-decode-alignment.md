# LM 解码阶段数值对照（`base_lm` / `residual_lm`）

目的：把自回归阶段的漂移拆成「LM 本身」和「采样器」两部分。

## 工具

- Rust：`VOXCPM_DUMP_AR=<dir>`（voxcpm-rs `468000c`）。写出 `step_m1_{lm,residual}_hidden`
  （预填充后）、`step_{i}_{lm_hidden,residual_hidden,pred_feat,dit_input}`，裸 f32 LE + `.shape`。
- PyTorch：`/tmp/ref_ar_dump.py`，复刻零样本 `_inference`，同参数同 seed。
- 配置：`Hello world, this is a test.`，`--timesteps 4 --seed 42 --max-len 8`，`min_len=2`，8 步。

## 关键结论：必须先消掉采样器的随机性

不加控制时，Rust CPU 与 PyTorch 的预填充一致性是 **43.84 / 44.54 dB**，但第一步就掉到
**1.97 / 4.39 dB**，`pred_feat` 是负 SNR。这不是累积误差，而是**第一次迭代就有结构差异**——
两侧 CFM 的初始噪声来自不同的随机流（Rust 用 `StdRng::seed_from_u64(42)`，PyTorch 用
`torch.manual_seed`），而 `pred_feat` 又直接喂回 LM，差异立刻放大。

两侧都强制 `z = 0`（Rust 用 `VOXCPM_Z_ZERO=1`，PyTorch 侧临时把 `torch.randn`/
`randn_like` 换成 zeros）后：

| 张量 | 结果（rust-cpu vs pytorch） |
|------|----------------------------|
| `step_m1_lm_hidden` | 43.84 dB |
| `step_0_lm_hidden` | 30.91 dB |
| `step_0_pred_feat` | 27.48 dB |
| `step_0_residual_hidden` | 40.61 dB |
| step 1–7 各张量 | 26.7 – 34.1 dB |

即：**LM 解码实现与参考一致到 27–40 dB**，之前的低 SNR 完全是采样随机性造成的假象。

## 另一个发现：f16 的自回归不稳定（未定论）

同一版本的 Vulkan 二进制，**改动前后**跑同参数得到完全不同的预填充结果：

| 运行 | `step_m1_lm_hidden` rms | 与 CPU 的 corr |
|------|------------------------|----------------|
| 旧二进制 | 0.5327 | **1.000000** |
| 新二进制 | 1.6872 | 0.099 |
| 回退到 `468000c` 重建 | 1.6872 | 0.099 |

回退后仍是 1.6872，说明**不是代码改动引起的**，而是「添加 `dit_input` dump」那次构建之后
首次复现、且稳定复现。预填充阶段完全不受 CFM 影响，出现 3.17 倍量级差异说明 f16 路径
存在与构建/驱动状态相关的非确定性（可疑：cubecl 的 kernel 选择或 autotune 缓存）。

这个现象使 **f16 的自回归一致性目前不可信**，也让之前「f16 vs CPU 逐 token 23–30 dB」
的结论需要重新验证（那次用的是旧二进制）。

## 待查

1. f16 的构建/运行非确定性：是否有 autotune 缓存（`CUBECL_CACHE`/`~/.cache/cubecl`），
   清理后是否回到 0.5327。
2. 若确认存在缓存依赖，需要固定编译选项再重跑 f16 的逐 token 对照。
3. `dit_input` dump 刚加上，尚未在 PyTorch 侧对齐（参考脚本还没有这一步）。
