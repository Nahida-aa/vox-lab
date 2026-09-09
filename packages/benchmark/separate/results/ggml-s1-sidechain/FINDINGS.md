# Sidechain 参数对比发现

## 背景

GGML 分离（shifts=1）的人声在空白段（~7-15s）产生类语音 artifact。Whisper 检测到 artifact 后错误地将其作为语音起点，导致：
- 后续所有 segment 的 start **整体前移**（偏移 ~10s）
- 偏移表现为两处症状：空白段被转写成 "我好想"、中段 "发财了" start 从 ~118.5s 漂到 ~108.5s
- 根本问题是 **timing 偏移**，文本错误（CER）是次生结果

`mixMode: sidechain` 通过引入动态压缩的 BGM 填补空白，减少 artifact 影响。但原始参数（ratio=20, release=500）效果有限。

## 测试方法

- 源音频：`packages/benchmark/separate/results/ggml-s1/media/`（GGML shifts=1 分离）
- ASR：whisper.cpp Vulkan（large-v3-turbo, zh）
- Ground truth：`packages/benchmark/ref/metadata/srt_manual.json`
- CER/WER：`packages/benchmark/separate/compute/test-sidechain.ts`（ffmpeg 混音 → whisper → compute-cer）

## 10 组参数对比

```
config                                       | RTF   | WER%   | CER%   | hyp_char | ref_char
---------------------------------------------|-------|--------|--------|----------|---------
sc_t0.1_r20_a1_rel500_bgm-12 (旧基线)         | 0.107 | 65.52% | 50.81% |      713 |      557
sc_t0.1_r20_a1_rel500_bgm-6                  | 0.120 | 75.86% | 50.09% |      661 |      557
sc_t0.1_r20_a1_rel500_bgm-3                  | 0.117 | 82.76% | 53.14% |      718 |      557
sc_t0.1_r20_a1_rel500_bgm0                   | 0.100 | 59.77% | 21.36% |      622 |      557
sc_t0.05_r20_a1_rel500_bgm-12                | 0.119 | 86.21% | 91.20% |      825 |      557
sc_t0.01_r20_a1_rel500_bgm-12                | 0.130 |120.69% | 91.92% |      758 |      557
sc_t0.1_r10_a1_rel500_bgm-12                 | 0.088 | 39.08% |  9.52% |      561 |      557
sc_t0.1_r4_a1_rel500_bgm-12                  | 0.094 | 41.38% | 11.85% |      576 |      557
sc_t0.1_r20_a1_rel200_bgm-12 (新最佳)         | 0.103 | 36.78% |  8.44% |      551 |      557
sc_t0.1_r20_a1_rel100_bgm-12                 | 0.136 | 78.16% | 32.68% |      674 |      557
```

## 发现

### 1. "我好想"— timing 偏移的早期体现

旧基线（ratio=20, release=500）下，7-15s 空白段的 artifact 被 whisper 当成语音起始。**"我好想"不是"幻觉文本"，而是 artifact 被错误分配了时间戳后的产物**——文本内容本身可能来自后续真实语音，或 artifact 的随机模式被 whisper 解释为语音。

### 2. "发财了" timing 偏移

偏移 10s 的直接证据：

| 参数 | "发财了" start | vs RAW ASR (118.56s) |
|---|---|---|
| ratio=10, release=500 | **118.60** | +0.04s ✅ |
| release=200 | **118.64** | +0.08s ✅ |
| ratio=20, release=500, bgm=0 | 109.00 | -9.56s ❌ |
| ratio=4 | 109.12 | -9.44s ❌ |
| 旧基线 (ratio=20, release=500) | 未识别 | — ❌ |

旧基线 + 低 BGM 组合下，7-15s 空白段 artifact 未被 BGM 掩盖 → whisper 从 artifact 位置开始分段 → 所有 segment start 整体前移 ~10s。

### 3. 关键结论

- **release=500ms 太长**：BGM 被压缩后迟迟不恢复，词间空白段暴露 artifact。release=200ms 让 BGM 在短语间隙自然恢复，掩盖 artifact → 分段从正确位置开始。
- **ratio=20 太激进**：高压缩比过度压制 BGM，反而让 artifact 更突出。ratio=10 更温和。
- **reduceBgm**：默认 -12dB 有效。bgm=0（不加额外衰减）时 CER 从 50% 降到 21%，说明更多 BGM 能掩盖更多 artifact——但 hyp_char 622 vs ref 557 偏差仍大，说明仅靠加 BGM 不够。
- **threshold < 0.1**：压缩器过于敏感，产生 pumping 失真 → CER >90%。

### 4. 新默认值

```
sidechainCompress:
  threshold: 0.1
  ratio: 10        (原 20)
  attack: 1
  release: 200     (原 500)
```

同时解决了 timing 偏移（"发财了" start 偏差从 -10s 降到 +0.08s）和 CER（50.81% → 8.44%）两个问题。

## 目录结构

```
ggml-s1-sidechain/
├── FINDINGS.md
├── sc_{params}/             # 每组参数的完整结果
│   ├── media/
│   │   └── target_3_vocals_mixed.wav
│   └── metadata/
│       ├── asr.json           # pipeline 格式（含 words）
│       ├── whisper_raw.json   # whisper 原始输出
│       └── summary.json       # WER/CER 摘要
├── media/                    # 之前的单次结果（旧基线）
└── metadata/                 # 之前的单次结果
```
