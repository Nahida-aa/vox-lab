# Whisper（ASR）研究记录

语音识别链路的实测与结论。既有详细记录保留在
[`packages/research/`](../../../packages/research/)，本文件是入口索引 +
跨阶段的结论汇总，新内容直接写在这里。

## 当前可用路径

| 模型 | 设备 | RTF | 说明 |
|------|------|-----|------|
| whisper.cpp（**Vulkan 构建**，二进制名 `whisper-cli`） | Vulkan (RADV) | **0.081–0.11** | large-v3-turbo，参数相关；预构建资产见 `release-whisper-{linux,windows}.yml` |
| faster-whisper（CTranslate2） | GPU (ROCm) | ~0.12 | 需 `HSA_OVERRIDE_GFX_VERSION=11.0.0`；`word_timestamps` 必须关闭 |
| faster-whisper | CPU | ~0.25 | large-v3-turbo int8，无 hang 风险，生产兜底 |
| openai-whisper（PyTorch） | CPU | — | 稳定可用 |

## 失败路径

| 路径 | 现象 |
|------|------|
| openai-whisper (PyTorch) + ROCm GPU | 100% SIGSEGV。25 次子进程试验 0 通过，连 `tiny`（39M）也立刻崩；`word_timestamps=False` 无效 |
| whisper.cpp 的 **HIPBLAS 构建** | 长音频（170 s）MES hang；短音频可跑，RTF ~0.36。注意这是同一份源码的另一个后端，与下面可用的 Vulkan 构建无关 |
| faster-whisper + `word_timestamps=True` | 第二遍对齐触发 GPU watchdog |

细节见 [whisper-rocm-hang.md](../../../packages/research/whisper-rocm-hang.md)。

## 关键发现

### 0. 「whisper.cpp 挂了」要先问是哪个构建

`whisper-cli` 只是 whisper.cpp 的可执行文件名，同一份源码可以按后端编出
Vulkan / HIPBLAS / CUDA / CPU 等不同版本。本机上：

- Vulkan 构建：稳定，RTF 0.081
- HIPBLAS 构建：短音频可跑（RTF ~0.36），长音频 MES hang

所以「whisper.cpp 会 hang」这个说法不准确，应该说「whisper.cpp 的 HIPBLAS 构建
在长音频上 hang」。

### 1. 同机上的反例，说明核显 Vulkan 并不慢

whisper.cpp Vulkan 转录 RTF **0.081**，比 CPU 快得多。这与 demucs 在 Burn wgpu
上跑不动（RTF 2.41）形成对照——同一块 780M，差别在实现而非硬件。

这也是 [demucs 研究](../demucs/README.md) 里「瓶颈在通用栈而非模型」的一条外部证据。

### 2. GPU Hang 会污染后续运行

一旦发生 GPU Hang（无论来自哪个原因），ROCm 显卡进入降级状态，**其他进程**的
GPU 操作也会连锁 hang，必须等恢复（`torch.cuda.device_count()` 探测或短暂等待）。

因此实验必须用「每个 trial 一个全新子进程」的方式做，且要能识别 exit code
`-11`（SIGSEGV）/ `-6`（SIGABRT）/ timeout。

### 3. 时间戳偏移不等于对齐错误

评测时 start/end 出现固定偏移，通常是 ASR 吞掉了填充词（`啊`/`嗯`/`哎`/笑声）
或边界低音量语音，而非时间戳算错。判断依据是 `missed` 段是否只含填充词。

### 4. sidechain 音频对 VAD 参数敏感的路径很关键

对 `temp-*` 参数，不做 sidechain 处理会让分段数骤降（86 → 48）、时间不可用。
详见 [`packages/research/asr/readme.md`](../../../packages/research/asr/readme.md) 的
normCER / 覆盖率对照表（最佳：`vad-v6-th02` + sidechain，normCER 8.41%）。

## 环境

- CPU：Ryzen 7 7840HS
- GPU：Radeon 780M（gfx1103，RADV）
- whisper.cpp 钉在 LocalDub 同 commit（Vulkan 后端）
- 评测脚本：`packages/benchmark/ref/compute/eval-ocr.ts`（统一文本归一化规则）

## 复现方式

```bash
# whisper.cpp 的 Vulkan 构建（可执行文件名 whisper-cli）
./build/bin/whisper-cli -m <ggml-large-v3-turbo.bin> -f input.wav

# PyTorch 路径的 hang 复现（每个 trial 独立子进程）
.venv/bin/python packages/research/whisper_rocm_hang/test_01_encoder.py
.venv/bin/python packages/research/whisper_rocm_hang/test_02_transcribe.py
```

## 未解决

- 曾观察到一次 PyTorch Whisper GPU 成功（2026-06-04），与 0/25 的失败数据矛盾，
  触发条件未知。
- whisper.cpp Vulkan 的 RTF 随参数波动（0.081–0.11），尚未做参数敏感性分析。
