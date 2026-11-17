# VoxCPM 研究记录

VoxCPM2 TTS 链路（MiniCPM4 LM + 残差 LM + CFM/DiT + AudioVAE）的实测与结论。
决策见 [ADR 0011](../../decisions/0011-voxcpm-gpu-backend-outlook.md)。

## 模型事实（`data/models/voxcpm2`）

| 项 | 值 |
|----|-----|
| 参数量 | 约 2.18 B（`model.safetensors` 4.4 GB，单文件 577 张量，BF16） |
| 结构 | `base_lm` 1472M（MiniCPM4，28 层 / hidden 2048 / GQA 2 kv）+ `residual_lm` 378M + `feat_encoder`(LocEnc) 208M + `feat_decoder`(UnifiedCFM/DiT) 212M + `audio_vae` 94M |
| 采样率 | 48 kHz |
| 生成方式 | 自回归：每 patch 一次 LM 前向 + CFM 多步去噪 + VAE 解码 |

## 环境与踩坑

两个 Python 环境并存（`.gitignore` 用 `.venv*/`）：

| 环境 | 内容 | 结论 |
|------|------|------|
| `.venv` | Python 3.14 + `torch 2.14.1+rocm7.2` | 模型能加载完，**生成阶段约 26 s 后进程被杀**；CPU/GPU 皆然 |
| `.venv312` | Python 3.12 + `torch 2.14.1+cpu` | 正常，所有实验在此完成 |

其他坑：

- `torchaudio` 没有 cp314 wheel（PyPI 与 rocm7.2 索引都没有），`voxcpm` 顶层无条件
  import → `core.py` 里加了桩（带 `__spec__`，否则 transformers 的可用性探测报错）。
- `dora-search` 在 py3.14 装不上，推断路径不需要，用注入的桩模块绕过。
- dtype 在 `config.json` 里，且**构造时**决定 KV cache 精度；事后 `.to()` 无效。
  做法是把模型目录用符号链接镜像一份、改写 `config.json` 的 `dtype` 再加载。

## PyTorch 分段耗时（`.venv312`，CPU）

`--max-len 32 --dit-steps 4`，墙钟 23.86 s / 音频 2.88 s → **RTF 8.29**。
计时方式：对叶子模块注册 forward hook，按顶层模块名归组。

| 阶段 | 耗时 | 占比 |
|------|------|------|
| `base_lm` | 4793 ms | 36.2% |
| `feat_decoder`（CFM/DiT） | 3921 ms | 29.6% |
| `audio_vae` | 1794 ms | 13.6% |
| `residual_lm` | 1370 ms | 10.4% |
| `feat_encoder` | 1249 ms | 9.4% |
| 投影层合计 | ~100 ms | 0.8% |
| 未归因（Python 循环、KV cache、采样） | 10633 ms | 44.6% |

**要点**：LM 合计 46.6%，CFM + VAE + encoder 合计 52.6%。后三段是并行、批量、
卷积/注意力密集的形状，适合搬 GPU；不必全栈重写。

## Burn 路径实测（`submodule/voxcpm-rs`）

标准文本 "Hello world, this is a test."，`--timesteps 4`（`VOXCPM_TIMESTEPS`）。

| 后端 | 加载 | 生成 | RTF | 结果 |
|------|------|------|-----|------|
| `cpu`（ndarray） | 65.7 s | 270.5 s | 338.1 | 可用 |
| `cpu-blas`（OpenBLAS 0.3.34） | 71.6 s | 314.2 s | 392.7 | 加 BLAS 反而更慢 |
| `vulkan`（bf16） | 58.9 s | — | — | panic：`cubek-matmul` 报 BF16 not supported |
| `vulkan`（f16） | 57.3 s | 3.57–15.9 s | **3.9–5.6** | **可用** |
| `wgpu`（f32） | 96.9 s | — | — | `radv/amdgpu: Not enough memory for command submission` |

### gfx1103 的 bf16 问题与绕法

本机 `vulkaninfo` 只有 `shaderFloat16 = true`，没有 `VK_KHR_shader_bfloat16`；
`cubek-matmul` 因此报 `BF16 lhs/rhs/output not supported`。

绕法是换浮点类型：`cli/src/bin/voxcpm-burn-vulkan.rs` 里按 `VOXLAB_FLOAT=f16|f32`
（默认 bf16）选择 `Vulkan<half::f16, i32>` 等。f16 与 bf16 同为 2 字节权重，
容量不变，实测可跑且比 PyTorch CPU 快约 2 倍。

wgpu f32 则受容量限制：4.4 GB bf16 → 8.7 GB f32，核显显存来自共享内存，放不下。

### Burn Vulkan 的确定性

`voxcpm-rs` 有两处无种子随机：

- `src/locdit/unified_cfm.rs:51`：CFM 初始噪声 `z ~ N(0,1)`
- `src/audiovae/layers.rs:177`：VAE `NoiseBlock` 的 `randn`

因此**同一文本多次生成的长度与内容都不同**（实测 0.64 / 2.88 / 2.56 s），
与 f16 无关——CPU 路径同样如此。

上游预留了 `VOXCPM_Z_ZERO=1`（把 CFM 的 `z` 置零）：

```
Got 192000 samples (4.00s) in 15.85s  RTF 3.96
Got 192000 samples (4.00s) in 15.59s  RTF 3.90
corr 1.0  maxdiff 0.0
```

可复现，但输出电平异常低（rms 0.0035 / 峰值 0.034，正常语音约 0.1–0.7），
疑似零初值下 CFM 退化，且 VAE 的 `NoiseBlock` 仍随机。**这是诊断手段，不是生产方案**。

### 已实现：`GenerateOptions::seed`

两处随机都改为接受 `Option<u64>`：

- `UnifiedCfm::forward(..., seed)` —— 用 `StdRng::seed_from_u64` + Box–Muller 生成
  `z ~ N(0,1)`
- `NoiseBlock::forward_seeded(x, seed)` —— 同上（附带 `CausalDecoder::forward_seeded`、
  `AudioVae::decode_seeded` 透传链）

对外暴露为 `GenerateOptions::seed`（`builder().seed(n)`）与 CLI `--seed`。
未播种时行为与上游完全一致（每次新的随机噪声）。

实测（Vulkan f16，`--timesteps 4`）：

| 输入 | 结果 |
|------|------|
| 英文 + `--seed 42` 跑两次 | 长度均 1.92 s，**corr 1.0 / maxdiff 0.0** |
| 英文 + `--seed 7` | 长度同样 1.92 s，但与 seed 42 的 corr 仅 −0.039（内容确实不同） |
| 中文 + `--seed 42` 跑两次 | 长度均 29.60 s，maxdiff 0.0 |

注意：长度稳定不等于内容正确。`你好。` 配上种子与 f16 会一直生成到 `max_len`
上限（29.6 s），换稍长的句子 `这是一段用于测试的中文语音。` 则正常（3.04 s）。

## stop head 排查（2026-11-17）

加了 `VOXCPM_DEBUG_STOP=1` 打印每步 stop logits（`DitStep` 内，`voxcpm-rs`
`a6b4a5c` 之后）。两组同参数（f16 / `--seed 42` / `--timesteps 4` / `--max-len 40`）对比：

**英文 `Hello world, this is a test.`** —— 正常停止（1.92 s）

```
[stop] logits=[29.609, -37.812] argmax=[false]   ← 预填充步，类 0 占优
[stop] logits=[7.0625, -10.07]  argmax=[false]
...
[stop] logits=[0.603, -0.682]   argmax=[false]   ← 中段类 1 逐步逼近
[stop] logits=[-2.789, 2.074]   argmax=[true]    ← 最终翻到类 1
```

**中文 `你好。`** —— 不停止，跑满 40 个 patch

```
[stop] logits=[29.594, -37.344] argmax=[false]
[stop] logits=[1.796, -2.668]   argmax=[false]
[stop] logits=[0.459, -0.460]   argmax=[false]   ← 两值几乎相等但对调
[stop] logits=[0.926, -0.883]   argmax=[false]   ← 之后稳定在类 0 略高
...
[stop] logits=[1.20, -1.38]     argmax=[false]
```

**结论与排除项**

- 不是 `argmax` 阈值问题：中文那组两类 logits 长期贴近 0（|logit| < 1.5），
  但始终类 0 略高，从未翻面；英文那组最后明确翻到类 1（+2.07 vs −2.79）。
- 不是死循环：中文组 logits 有起伏（0.38 → 1.23），模型在持续产生内容，只是
  stop head 一直判「继续」。语音内容本身是否重复未验证。
- 不是纯精度问题，也无法完全排除：`f32` 对照跑不起来（权重 8.7 GB，780M
  共享显存 OOM），这一项仍待有更大显存的机器验证。
- 不是实现与人不同：Rust 的 `stop_head(silu(stop_proj(lm_hidden)))` 与
  PyTorch 参考（`model/voxcpm2.py:1113`）的时序、取用的 hidden 完全一致。
- **参考实现本身也不稳定**：PyTorch 同 seed 连跑 3 次「你好。」，音频长度分别是
  2.40 / 0.64 / 0.64 s —— 说明 `torch.manual_seed` 没覆盖全部随机源（我们只控制了
  CFM 的 z），短文本的停止点本身就随机。

因此当前判断：**短中文文本停不下来是模型/采样层面的脆弱性，不是 Rust 移植缺陷**。
后续要做确定性评测，应使用较长、结构完整的句子（实测 `这是一段用于测试的中文语音。`
稳定在 3.04 s）。

## 结论

1. **能跑通的可用路径**：PyTorch CPU（RTF 8.29）、Burn Vulkan f16（RTF 3.9–5.6）。
   Burn CPU（338）和 Burn wgpu f32（OOM）不可用。
2. **GPU 化的正确范围**：只搬 CFM + VAE + encoder（52.6% 时间、约 514 MB 权重），
   LM 留 CPU。全栈重写收益不大，且受带宽限制。
3. **确定性缺失**：Burn 侧需要给 `GenerateOptions` 加 seed（两处随机都用它），
   否则无法回归测试。

## 仓库内实现

| 位置 | 说明 |
|------|------|
| `packages/voxcpm_pytorch` | `core`（加载/生成/计时/dtype 实验）、`cli`（generate/profile/dtype）、`server`（Gradio 复刻云端签名） |
| `submodule/voxcpm-rs` | 独立 workspace（bf16 补丁需要 `[patch.crates-io]` 在根生效） |
| `submodule/voxcpm-rs/cli` | 三个后端 bin：`voxcpm-burn-{cpu,wgpu,vulkan}` |

## 复现方式

```bash
# PyTorch 分段耗时
cd packages && ../.venv312/bin/python -m voxcpm_pytorch.cli profile \
  --model-dir /home/aa/repos/env_ls/LocalDub/data/models/voxcpm2 \
  --device cpu --max-len 32 --dit-steps 4

# Burn Vulkan f16
cd submodule/voxcpm-rs
cargo build --release -p cli --no-default-features --features vulkan --bin voxcpm-burn-vulkan
AMD_VULKAN_ICD=RADV VOXLAB_FLOAT=f16 ./target/release/voxcpm-burn-vulkan \
  --model-dir <模型目录> --timesteps 4 "Hello world, this is a test." /tmp/out.wav
```
