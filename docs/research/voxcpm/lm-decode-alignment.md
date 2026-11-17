# LM 解码阶段数值对照（`base_lm` / `residual_lm`）

目的：把自回归阶段的漂移拆成「LM 本身」和「采样器」两部分。做法是给 Rust 侧加
逐 token dump，然后在 PyTorch 参考里复刻同样的循环，同样 dump，逐张量比 SNR。

## 工具

- Rust：`VOXCPM_DUMP_AR=<dir>`（voxcpm-rs `468000c`）。每步写出：
  `step_m1_{lm,residual}_hidden`（预填充后、第一次 DiT 的输入）、
  `step_{i}_{lm_hidden,residual_hidden,pred_feat}`（`lm_step` 之后的下一步输入与 DiT 输出）。
  格式是裸 f32 LE + 同名 `.shape`。
- PyTorch：`/tmp/ref_ar_dump.py`，复刻零样本路径的 `_inference`（不经过 `_generate`
  的 badcase 重试），同参数、同 seed。

## 配置

`Hello world, this is a test.`，`--timesteps 4 --seed 42 --max-len 8`，`min_len=2`，
两侧都跑 8 步（未触发 stop）。

## 结果

### Vulkan f16 vs Rust CPU（同一份实现，只差精度）

| 张量 | step 0 | step 3 | step 7 |
|------|--------|--------|--------|
| `lm_hidden` | 41.84 dB | 27.17 dB | 30.23 dB |
| `pred_feat` | 56.02 dB | 43.54 dB | 23.55 dB |
| `residual_hidden` | 56.94 dB | 39.32 dB | 26.75 dB |
| 预填充 (`step_m1`) | 57.50 / 60.16 dB | | |

**结论**：同一实现内，f16 的误差逐 token 缓慢累积（42 → 30 dB），8 步后仍在 23–30 dB，
没有爆炸。这是纯粹的精度代价，可接受性取决于应用。

### Rust CPU vs PyTorch 参考

| 张量 | 预填充 | step 0 | step 7 |
|------|--------|--------|--------|
| `lm_hidden` | **43.84 dB** | **1.97 dB** | 11.41 dB |
| `residual_hidden` | **44.54 dB** | **4.39 dB** | −0.35 dB |
| `pred_feat` | — | **−1.93 dB** | −2.97 dB |

**这里出现断崖**：预填充阶段两侧高度一致（44 dB），但第一次自回归步就掉到 2–4 dB，
`pred_feat` 更是负 SNR。说明不是「累积误差」，而是**第一次迭代就有结构差异**。

## 待查

1. **KV cache 的 step 位点**：PyTorch 用 `kv_cache.step()` 返回的整数做位置索引，
   Rust 用 `state.steps_taken`。两者若差一位，attention 位置编码就会整体错开。
2. **DiT 条件输入**：参考实现用 `prefix_feat_cond.transpose(1, 2)`，Rust 用
   `state.prefix_feat_cond.swap_dims(1, 2)`；预填充阶段两者一致，但 `lm_step` 之后
   的更新时机需要逐行核对。
3. **`fsq_layer` 的作用点**：参考实现里 `lm_hidden = self.fsq_layer(lm_hidden)` 在
   `base_lm.forward_step` 之后；Rust 在 `lm_step` 内部，两者对 `residual_lm` 的输入
   是否等价需确认。
4. **噪声来源**：`pred_feat` 由 CFM 采样得到，参考脚本没有插 CFM 的 seed（只靠
   `torch.manual_seed(42)`），而 Rust 用的是 `StdRng::seed_from_u64(42)` —— 两边的
   随机流本来就不同。这一条足以单独解释 `pred_feat` 的负 SNR，也必然会传导到
   后续的 `lm_hidden`。

第 4 条最可能是主因。修正方法：给参考脚本注入与 Rust 相同的 z（把 Rust dump 的
初始噪声喂给 PyTorch 的 `feat_decoder`），或者在两侧都用 `VOXCPM_Z_ZERO=1` 做
纯确定性对比。
