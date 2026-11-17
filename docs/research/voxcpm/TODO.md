# VoxCPM 待办

研究与实测记录见 [README.md](./README.md) / [burn-gpu-backends.md](./burn-gpu-backends.md) /
[f16-precision.md](./f16-precision.md)。本文件只列「接下来做什么、为什么」。

## 1. dump LM 解码阶段的中间张量

**为什么**：`base_lm` + `residual_lm` 占 PyTorch 总时长 46.6%，是 f16 影响最可能暴露
的地方，也是目前唯一没有数值对照的阶段。

**怎么做**：给 `voxlab-dump` 增加 token 级 trace（每步导出 lm_hidden / residual_hidden），
用相同的预填充输入与 PyTorch 参考逐 token 比对 SNR。

**前置**：需要固定的 token 序列——用 `--seed` + 固定 `max_len` 保证两侧走同样的步数。

## 2. 端到端 f16 质量评估（多 seed 平均）

**为什么**：目前只测了 VAE 编码（29.40 dB）和特征编码（42.76 dB），没有端到端指标。
单次波形对比不可靠——采样本身发散，同 seed 下 PyTorch 侧长度也会变（2.40 / 0.64 / 0.64 s）。

**怎么做**：N 个种子 × 固定文本，分别跑 f16 Vulkan 与 PyTorch bf16，统计 SI-SDR 或
等价指标；同时记录长度分布。

**依赖**：第 1 项完成后才有意义（否则无法区分是 LM 还是 VAE 的误差）。

## 3. 排查短文本 stop head 不触发

**现状**：`你好。` + f16 + `--seed 42` 跑满 `max_len`；stop logits 长期在 ±1.5 内且类 0
略高，从未翻面。英文短句和稍长中文都正常（3.04 s）。

**待查**：
- f32 下是否同样（需要显存 > 8.7 GB 的机器；780M 只有 4 GB VRAM）
- 是否与文本长度/预填充长度有关（`min_len=2` 之外的判定）
- 采样发散时 stop head 的输入分布是否本身就不稳定

## 4. GPU 化 CFM + VAE（自研 WGSL）

**为什么**：PyTorch 分段显示 `feat_decoder`(CFM) 29.6% + `audio_vae` 13.6% +
`feat_encoder` 9.4% = **52.6%**，权重合计约 514 MB（f32 约 1 GB），容量上完全可行。
LM 留在 CPU 也不会成为瓶颈。

**怎么做**：kernel 骨架可复用 `packages/demucs-wgpu`；先做 CFM，再做 VAE。

**风险**：f16 在 VAE 上有 3.5% 相对误差，自研 kernel 若用 f32 可绕开该问题（容量允许）。
