# f16 精度评估（Vulkan / 780M）

目标：确认 `Vulkan<half::f16>` 相对 PyTorch bf16 参考的数值损失。方案是逐阶段
dump 中间张量做 SNR 对比，避开自回归采样带来的发散。

## 工具

`cli/src/bin/voxlab-dump`（voxcpm-rs `59d3a66`）——为数值对照新增，不属于上游公开 API。

```bash
voxlab-dump --model-dir <dir> --out <f32le> <audio|encode|vae|generate> [参数]
```

- `audio`：原始单声道音频 `[B,1,T]`（VAE 编码器要求 16 kHz）→ latent `[B,64,T']`
- `encode`：音频 patch `[B,T,P=4,D=64]` → LM hidden `[B,T,2048]`
- `vae`：latent `[B,D,T]` → 波形（带固定 seed 的 VAE 噪声）
- `generate`：完整生成（受 `--seed` 控制）

配套方法：`VoxCPM::encode_audio_f32` / `encode_feat_f32` / `decode_feat_f32`，
以及 PyTorch 侧的 `ref_dump.py`（同样输出裸 f32 LE）。

## 结果（同输入，逐阶段）

| 阶段 | 对比 | corr | SNR | 结论 |
|------|------|------|-----|------|
| VAE 编码 | rust-cpu vs PyTorch | 1.000000 | **90.35 dB** | 移植等价 |
| VAE 编码 | **vulkan-f16 vs PyTorch** | 0.999426 | **29.40 dB** | 有可测损失 |
| 特征编码 | rust-cpu vs PyTorch | 0.999973 | 42.73 dB | 仅数值噪声 |
| 特征编码 | **vulkan-f16 vs PyTorch** | 0.999973 | **42.76 dB** | **f16 未带来额外损失** |

关键对照：rust-cpu 与 vulkan-f16 在特征编码上的 SNR 几乎相同（42.73 vs 42.76），
说明这一阶段的差距来自实现细节（f32 与 bf16 参考的累加顺序），不是 f16。
而 VAE 编码阶段 f16 掉到 29.40 dB（corr 0.9994，max|d| 0.48，latent rms ≈ 0.82），
是 f16 尾数不足的真实代价。

### 精度解读

- 29 dB 的 SNR 在潜变量上意味着约 3.5% 的相对误差。对会被解码成波形的 VAE latent，
  是否可接受需要听感验证——**本次没做**。
- 特征编码 42 dB 属于正常浮点噪声水平，不会影响 LM 上下文。

## 未完成

- 没有做端到端听感/客观指标（如 SI-SDR）对比：自回归采样本身发散，单次对比不可靠，
  需要多 seed 平均。
- LM 解码阶段的逐 token 对照已补上，见 [lm-decode-alignment.md](./lm-decode-alignment.md)：
  在 `z = 0` 控制下 rust-cpu 与 PyTorch 一致到 26.7–40.6 dB。但该结论基于当时的构建，
  而 Vulkan 后端已被发现存在构建间非确定性（见 [TODO.md](./TODO.md) 第 0 项），
  f16 侧的逐 token 数字需要在固定构建下重验。
- f32 参考在 780M 上无法运行（8.7 GB 权重 vs 4 GB VRAM + GTT），所以 f16 与 f32 的差别
  只能通过 CPU f32（`rust-cpu`）间接比较。
