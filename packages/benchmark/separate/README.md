# Separate Benchmark — Demucs 音频分离

## 模型参数

## 测试内容
对三段不同时长（10s / 60s / 120s）的音频（粉红噪声，16-bit 44.1kHz 单声道 WAV）进行 Demucs 4-stem 分离，提取人声 + 背景音。对比三个后端，支持 `shifts=1`（公平对比）和 `shifts=3`（高质量）：

| 后端 | 路径 | 模型加载方式 | 推理引擎 | 支持 shifts |
|------|------|------------|---------|------------|
| **PyTorch** | `scripts/separate/run.py` | 每次子进程从头加载 | PyTorch CPU | 1 / 3（通过 `--shifts`） |
| **ORT** | `Demucs` class (onnxruntime-node) | InferenceSession 创建 | onnxruntime-node CPU | 无 shift 机制 |
| **GGML** | `demucs_mt.cpp.main` (C++) | 每次子进程加载 | Eigen + OpenBLAS + OpenMP | 仅 shift=1（硬编码） |

**RTF 定义**：三个后端统一为 `processTimeS / durationS`，其中 `processTimeS = totalTimeS - loadTimeS`（不含模型加载时间）。

## 使用
```bash
# 跑全部后端
bun run packages/benchmark/separate/run-all.ts

# 仅 GGML（推荐先装 openblas）
OMP_NUM_THREADS=2 bun run packages/benchmark/separate/run-ggml.ts

# 仅 ONNX
bun run packages/benchmark/separate/run-onnx.ts

# 仅 PyTorch（默认 shifts=3）
bun run packages/benchmark/separate/run-pytorch.ts

# 视频实测：对比 GGML (shift=1) vs PyTorch (shifts=1/3)，保存 WAV
bun run packages/benchmark/separate/run-video.ts

# ORT 视频实测
bun run packages/benchmark/separate/run-video-onnx.ts

# WER/CER 质量对比（需 whisper-cli + ground truth）
bun run packages/benchmark/separate/compute/run-wer.ts
```

结果写入 `results/separate-bench.json`。

## 当前结果 (2026-06-09)

### 视频实测对比（`video_source.mp4` 170.1s 立体声） htdemucs_ft + whisper

以人工标注转录为 ground truth, whisper.cpp (large-v3-turbo) 对分离后 vocals 转写的准确率。整合速度与质量指标，含 LLM 纠错后结果（`gemma4:31b`）：

| 后端 | Shifts | RTF | CER | LLM后CER | s_off_mean | s_off_median | hyp/ref |
| --- | ---    |---  |---  |---       |---         |---           |---      |
| **无分离 (原始音频)** |  — | **0** | 8.08% | — | +0.09s | +0.04s | 556/557 |
| **ORT bag** (vocals-only) | 1 | **1.397** | **7.54%** | — |  |
| **ORT bag** (4-stems) | 1 | 6.290 | 7.54% | — | 全导出 |
| **GGML** | 1 | **1.362** | 8.80% | 4.85% | — |
| **GGML + sidechain** | 1 | 1.362 | 8.44% | — | +0.09s |
| **GGML + sidechain + temp-02** | 1 | 1.362 | 7.72% | — | +0.04s | | 557/557 |
| **GGML + sidechain vad-v6-th02** | 1 | 1.362 | 7.00% | — | -0.68s | | 552/557 |
| **PyTorch** | 1 | 2.647 | 7.72% | 4.49% | — |
| **PyTorch** | 3 | 7.922 | 7.18% | **3.41%** |  |

† 分离 RTF（与 GGML 同，sidechain 仅在 ASR 阶段增加 ffmpeg filter 开销，可忽略）。whisper.cpp Vulkan 转录 RTF **0.081**。

> **推荐：不分离 (原始视频, CER 8.08%) → GGML + sidechain (CER **7.00%**, chars 552/557 最佳 CER)** — sidechain compression 在空白段填入降幅背景音，消除 Demucs 分离低噪引起的 "我好想" 幻觉，同时不破坏有声段。whisper.cpp Vulkan (RTF **0.081**) 进一步降低分离+转录耗时。

- **LLM 纠错效果显著**：CER 降低 40-52%，且**分段映射正确**（`segments[].text` 逐条更新，不破坏完整性）
- **行号模式 + 全文上下文**：先提供完整对话供 LLM 理解语境，再要求逐行修正；兼顾 CER 改善与分段完整性。纯行号模式（无上下文）CER 改善仅 26-40%
- **GGML WAV 格式**：默认输出 32-bit float，whisper.cpp 对其转录质量会显著下降（CER 14.72%），需转 16-bit
- **ORT bag 架构**：从单模型 (`htdemucs_ft_vocals.onnx`) 切换为 4 个 per-stem specialist fp16weights ONNX 模型。`Demucs` 构造函数支持 `stems: Stem[]` 参数（默认 `['vocals']`），指定需要跑的目标可以大幅降低 RTF

### "我好想" 幻觉分析

"我好想" 幻觉仅出现在 **Demucs 分离后** 的 vocals 中，原始音频无此幻觉。不同 ASR 后端对同一段分离音频的幻觉跨度不同：

| 音频 \ ASR | whisper-cli | faster-whisper | whisper-pytorch |
|-----------|------------|---------------|----------------|
| **Raw** | ❌ 无 | ❌ 无 | ❌ 无 |
| **GGML** | ⚠️ 7-23s (16s) | ⚠️ 7-15.8s (8.8s) | ⚠️ 7-15.8s (8.8s) |
| **ORT bag** | ⚠️ 7-23s (16s) | ⚠️ 7-15.8s (8.8s) | ⚠️ 7-15.8s (8.8s) |
| **GGML + sidechain** | ✅ **无** | — | — |

whisper-cli 的 VAD 分段策略导致幻觉跨度最大 (16s)，而 faster-whisper 和 whisper-pytorch 的跨度一致 (8.8s)。根因是 Demucs 分离在 7-15.8s 区域产生了低噪声信号，各 ASR 后端的 VAD 对该噪声的敏感度不同。

**Sidechain compression 消除幻觉**：在 7-15.8s 空白段，强压缩后 BGM 以 -12dB 填充，掩盖了 Demucs 分离低噪，有效抑制幻觉产生。CER 从 8.44% (纯 vocals) 降至 **7.00%**，字符数 557 与 GT 完全一致。

### 多 ASR 后端对比（全文本 WER/CER）

在同一组 Demucs 分离 vocals 上对比三种 ASR 后端：

| 音频 \ ASR | whisper-cli (whisper.cpp) | faster-whisper (CPU) | whisper-pytorch (CPU) |
|-----------|--------------------------|---------------------|----------------------|
| **Raw** | CER **7.90%**, WER 42.53% | CER **14.72%**, WER 47.13% | CER **13.11%**, WER 100%† |
| **GGML** | CER **8.80%**, WER 47.13% | CER **11.49%**, WER 40.23% | CER **8.98%**, WER 100%† |
| **ORT bag** | CER **7.54%**, WER 40.23% | CER **10.05%**, WER 52.87% | CER **8.26%**, WER 100%† |

† whisper-pytorch 输出无空格分词导致 WER 被膨胀到 100%，CER 是有效指标

| ASR 后端 | 设备 | RTF 估算（CPU） | 质量排名 (CER) |
|---------|------|----------------|--------------|
| whisper-cli (whisper.cpp) | CPU (`-ng`，GPU MES hang) | ~1.31 (GPU 参考 ~0.36) | 🥇 最佳 |
| whisper-pytorch | CPU (GPU segfault) | ~1.55 | 🥈 |
| faster-whisper | CPU (GPU CTranslate2 不兼容 ROCm) | ~1.02 | 🥉 最差 |

结论：**whisper-cli (whisper.cpp)** 在质量和速度上都最优；faster-whisper 的 CER 在各音频源上均最差，且 GPU 不可用。whisper-pytorch CPU 质量居中但速度最慢。

> **GPU 稳定性**：whisper-cli (HIPBLAS)、whisper-pytorch、faster-whisper (CTranslate2) 三者的 GPU 加速在 Radeon 780M / ROCm 7.2.4 上均不可行 — GPU Hang 或 segfault。所有 benchmark 数据均在 CPU 上运行。whisper-cli 的 GPU 短音频参考 RTF ~0.36（仅稳定时测得，长音频 MES hang）。

### GGML 合成音频（OpenBLAS + MT，Ryzen 7 H 255 8c/16t）

| Engine | Device | Audio | Dur(s) | Load(s) | Proc(s) | RTF |
|--------|--------|-------|--------|---------|---------|-----|
| ggml | cpu | short | 10.0 | 0.2 | 28.8 | 2.878 |
| ggml | cpu | medium | 60.0 | 0.2 | 83.7 | 1.395 |
| ggml | cpu | long | 120.0 | 0.2 | 169.5 | 1.412 |
| ggml | cpu | video_source.mp4 | 170.1 | 0.2 | 235.6 | 1.386 |

**SMT 对比**（video_source.mp4 170.1s）：

| 配置 | 总线程 | Wall(s) | Proc(s) | RTF | Speedup |
|------|--------|---------|---------|-----|---------|
| mt=4, OMP=2 | 8 (物理核) | 235.4 | 235.4 | 1.384 | 1.000 |
| mt=4, OMP=4 | 16 (含 SMT) | 224.4 | 224.4 | 1.319 | 1.049 |

SMT 超线程对 Demucs 密集矩阵运算帮助有限（约 5%），这是因为 Demucs 的 Eigen 矩阵乘是**内存带宽瓶颈**，额外逻辑核共享 L2 cache 和 FPU，收益很小。`OMP_NUM_THREADS=2`（仅物理核）是性价比最优策略。

### ORT + PyTorch 合成音频（旧数据，归一化修正前）

**注意**：PyTorch 旧数据为 `shifts=3`（默认值），新数据需用 `--shifts` 重跑。ORT 视频 RTF 见上表。

| Engine | Device | Audio | Dur(s) | Load(s) | Proc(s) | RTF |
|--------|--------|-------|--------|---------|---------|-----|
| ort | cpu | short | 10.0 | 16.9 | 21.6 | 2.161 |
| ort | cpu | medium | 60.0 | 16.9 | 142.7 | 2.379 |
| ort | cpu | long | 120.0 | 16.9 | 258.8 | 2.157 |
| pytorch | cpu | short | 10.0 | 2.3 | 107.1 | 10.712 |
| pytorch | cpu | long | 120.0 | 2.3 | 1422.3 | 11.853 |

### 备注
- **PyTorch medium 旧数据失败**（91% 时出错），short & long 有效
- **加载时间**：GGML 0.2s vs ONNX 16.9s vs PyTorch 2.3s（GGML 是 C++ 裸加载最快；PyTorch 含 Python import 开销；ONNX 含 ONNX 文件验证 + session 创建）
- **GGML 外归一化**：`model_apply.cpp` 实现了和 `api.py` 一致的 mono mix → scalar mean/std，输出音量与 PyTorch 一致
- **GGML 线程策略**：mt=4 std::thread + OMP_NUM_THREADS=2 = 8 线程（物理核数），避免 SMT 竞争 L2 cache
- **Shifts 参数**：`--shifts N` 仅在 PyTorch 后端支持（`_engine.py` / `run.py`），GGML 硬编码 shift=1，ORT 无 shift 机制。`shifts=3` 是 PyTorch 默认值，`shifts=1` 用于公平对比
- **合成 vs 视频**：GGML 在视频（170s 立体声）上 RTF 1.362，与合成数据一致
- **GPU 稳定性**：gfx1103 (RDNA 3) 的 MES firmware 0x83 在长时间连续 GPU kernel 下会触发 `REMOVE_QUEUE` hang，导致 GPU reset。whisper-cli GPU 仅短音频稳定，长音频（170s）必然 hang。CPU 是唯一稳定路径

## ASR 分段人工标注对比

`compute/asr_seg_compare.ts` 将各后端 ASR 结果逐段与人工标注 (`packages/benchmark/ref/metadata/srt_manual.json`) 按时间对齐，统计：

| 指标 | 说明 |
|------|------|
| 匹配 | 能对齐到 GT 的 ASR 段数 |
| GT未匹配 | GT 中 ASR 完全漏掉的段 |
| 精确(原/修) | 文本与 GT 完全一致的段数 |
| LLM修正 | LLM 把错误改正确 |
| LLM加错 | LLM 把正确的改错 |
| 时间偏差 | ASR 段 start 与 GT 段 start 的平均差距 |

关键发现：
- **LLM 修正正确**：`尚寝→尚浅`, `拜剑师祖→拜见师祖`, `常跪→长跪`, `零食→灵石`
- **分段边界不匹配**是主要噪音来源：GT 89 段 vs ASR 89-93 段，时间偏差 ~0.8s。全文本 CER 比逐段对比更能反映 LLM 纠错质量

## 环境
- APU: AMD Ryzen 7 H 255 w/ Radeon 780M Graphics (8c/16t, Zen4)
- ROCm 7.2.4 (`/opt/rocm`)
- PyTorch 2.12.0+rocm7.2（从 CUDA 2.12.0+cu130 切换）
- onnxruntime-node 1.26.0
- openblas 0.3.33
- Demucs submodule (htdemucs_ft)
- demucs.cpp submodule (ggml + Eigen + OpenBLAS + OpenMP)
