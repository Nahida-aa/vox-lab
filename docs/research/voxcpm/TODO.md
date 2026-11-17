# VoxCPM 待办

研究与实测记录见 [README.md](./README.md) / [burn-gpu-backends.md](./burn-gpu-backends.md) /
[f16-precision.md](./f16-precision.md) / [lm-decode-alignment.md](./lm-decode-alignment.md)。
本文件只列「接下来做什么、为什么」。

## 已完成

- ✅ **dump LM 解码阶段**：`VOXCPM_DUMP_AR=<dir>` 逐 token 导出
  `step_m1_*` / `step_{i}_{lm_hidden,residual_hidden,pred_feat,dit_input}`。
  在 `z = 0` 控制下，LM 解码与 PyTorch 参考一致到 **27–40 dB**（此前 1.97 dB 的低值
  是采样随机性造成的假象，见 [lm-decode-alignment.md](./lm-decode-alignment.md)）。

## 0. 排查 Vulkan 后端的构建间非确定性（阻塞项）

**为什么**：同一份源码、同一份二进制连跑结果一致，但**重新构建后结果就变了**
（`step_m1_lm_hidden` rms 0.5327 ↔ 1.6872，与 CPU 的 corr 1.0 ↔ 0.099）。预填充阶段
不受 CFM 影响，且换 bf16 + naive matmul 同样出现，说明与精度无关，指向 cubecl 在
Vulkan 上的 kernel 选择（构建产物与驱动的交互）。`~/.cache/cubecl` 不存在，已排除
autotune 缓存。

**影响**：之前所有 f16 的逐 token 结论（23–30 dB）基于旧二进制，需要在新构建下重验。

**怎么做**：
- 固定构建（记录 git commit + 编译器版本），连做两次 clean build 对比是否稳定
- 对比两次构建的 SPIR-V / kernel 选择（cubecl 有没有可观测的日志）
- 若找到触发条件，把它固定进构建配置或记录成已知约束

## 1. 端到端 f16 质量评估（多 seed 平均）

**为什么**：目前只有分段指标（VAE 编码 29.40 dB、特征编码 42.76 dB、LM 解码 27–40 dB），
没有端到端波形质量。单次对比不可靠——采样本身发散。

**怎么做**：N 个种子 × 固定文本，分别跑 f16 Vulkan 与 PyTorch bf16，统计 SI-SDR 或
等价指标，同时记录长度分布。

**依赖**：第 0 项（否则同一构建内可复现、但构建一变结论就不可比）。

## 2. 排查短文本 stop head 不触发

**现状**：`你好。` + f16 + `--seed 42` 跑满 `max_len`；stop logits 长期在 ±1.5 内且类 0
略高，从未翻面。英文短句和稍长中文都正常（3.04 s）。

**待查**：
- f32 下是否同样（需要显存 > 8.7 GB 的机器；780M 只有 4 GB VRAM）
- 是否与文本长度/预填充长度有关（`min_len=2` 之外的判定）
- stop head 的输入（`lm_hidden`）在该组合下是否本身就偏离（与第 0 项相关）

## 3. 自研 WGSL 重写 CFM + VAE（可选方向，非既定方案）

**前提先说清**：整模型已经能在 Vulkan f16 上跑（RTF 3.9–5.6），所以这不是「必须搬 GPU」，
而是「要不要用自研 kernel 替掉 cubecl 这一层」。

**为什么可能值得做**：CFM + VAE + encoder 合计 52.6% 的 CPU 耗时、权重约 514 MB
（f32 约 1 GB），容量不构成障碍；`packages/demucs-wgpu` 提供了可复用的 kernel 骨架。
另一个动机是 cubecl 在 Vulkan 上的构建间非确定性（见第 0 项）——自研 kernel 可以绕开这一层。

**代价**：要自己维护 CFM / VAE 的全部算子；相比换精度、调 kernel 选择，工作量大一个量级。

**风险**：f16 在 VAE 上有 3.5% 相对误差；自研 kernel 若用 f32 可绕开（容量允许）。
