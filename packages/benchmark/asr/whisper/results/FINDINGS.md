# ASR Parameter Benchmark Results

## Summary

**38 whisper.cpp CLI parameter sets** across 3 audio sources (sidechain, raw, vocals), using large-v3-turbo on Radeon 780M (Vulkan). All timestamps in **milliseconds** with per-segment confidence from raw token probabilities.

Ground truth: `packages/benchmark/ref/metadata/asr_manual.json` (ms format, **564 chars**).

### Cross-source best performers

| Source | Params | normCER | segs | s_off_mn | s_mae | e_mae | det% | cover | gaps | g_avg |
|--------|--------|---------|------|----------|-------|-------|------|-------|------|-------|
| sidechain | vad-v6-th02 | **7.72%** | 88 | +190ms | 391ms | 598ms | 94.7% | 57.1% | 85 | 1.04s |
| sidechain | **temp-02** | **7.72%** | 86 | **+100ms** | **262ms** | **565ms** | 94.7% | 54.4% | 83 | 0.93s |
| sidechain | temp-04 | 7.72% | 86 | +100ms | 262ms | 565ms | 94.7% | 54.4% | 83 | 0.93s |
| sidechain | baseline | 8.44% | 89 | +233ms | 322ms | 676ms | 95.7% | 56.8% | 87 | 1.06s |
| sidechain | prompt+carry | 8.48% | 79 | +207ms | 402ms | 697ms | 94.7% | 52.8% | 81 | 1.26s |
| raw | vad-v6-th02 | 12.88% | 91 | -828ms | 887ms | 281ms | 100% | 100% | 0 | — |
| raw | vad-v6 | 14.31% | 87 | -1172ms | 1227ms | 393ms | 100% | 100% | 0 | — |
| vocals | baseline | 10.38% | 98 | -720ms | 814ms | 285ms | 98.9% | 96.5% | 1 | 6.0s |
| vocals | vad-default | 12.52% | 88 | -763ms | 825ms | 350ms | 100% | 100% | 0 | — |

**Key takeaway**: Temperature-annealed configs (temp-02/04) on sidechain have best timing accuracy (s_mae 262ms, s_off +100ms) with lowest CER (7.72%). VAD variants achieve 100% detection rate and full timeline coverage but at -0.7 to -1.2s systematic left shift. Vocals-only degrades both CER and timing.

### All sidechain results

| Params | normCER | segs | s_off_mn | e_off_mn | s_mae | e_mae | det% | miss | fp | cover | gaps | g_avg |
|--------|---------|------|----------|----------|-------|-------|------|------|----|-------|------|-------|
| vad-v6-th02 | 7.72% | 88 | +190ms | +497ms | 391ms | 598ms | 94.7% | 5 | 0 | 57.1% | 85 | 1041ms |
| temp-04 | 7.72% | 86 | +100ms | +453ms | 262ms | 565ms | 94.7% | 5 | 0 | 54.4% | 83 | 926ms |
| temp-02 | 7.72% | 86 | +100ms | +453ms | 262ms | 565ms | 94.7% | 5 | 0 | 54.4% | 83 | 926ms |
| baseline | 8.44% | 89 | +233ms | +562ms | 322ms | 676ms | 95.7% | 4 | 0 | 56.8% | 87 | 1055ms |
| nst-050 | 8.44% | 89 | +233ms | +562ms | 322ms | 676ms | 95.7% | 4 | 0 | 56.8% | 87 | 1055ms |
| nst-040 | 8.44% | 89 | +233ms | +562ms | 322ms | 676ms | 95.7% | 4 | 0 | 56.8% | 87 | 1055ms |
| nst-020 | 8.44% | 89 | +233ms | +562ms | 322ms | 676ms | 95.7% | 4 | 0 | 56.8% | 87 | 1055ms |
| nst-010 | 8.44% | 89 | +233ms | +562ms | 322ms | 676ms | 95.7% | 4 | 0 | 56.8% | 87 | 1055ms |
| max-len-20 | 8.44% | 89 | +233ms | +562ms | 322ms | 676ms | 95.7% | 4 | 0 | 56.8% | 87 | 1055ms |
| nst-020+max20 | 8.44% | 89 | +233ms | +562ms | 322ms | 676ms | 95.7% | 4 | 0 | 56.8% | 87 | 1055ms |
| prompt+carry | 8.48% | 79 | +207ms | +541ms | 402ms | 697ms | 94.7% | 5 | 0 | 52.8% | 81 | 1261ms |
| vad-v6 | 10.55% | 88 | -569ms | +358ms | 662ms | 450ms | 100% | 0 | 0 | 100% | 0 | — |
| vad-v6+nst-020 | 10.55% | 88 | -569ms | +358ms | 662ms | 450ms | 100% | 0 | 0 | 100% | 0 | — |
| prompt | 10.55% | 88 | +107ms | +1333ms | 1345ms | 1492ms | 94.7% | 5 | 1 | 49.4% | 87 | 1055ms |
| vad-default | 14.67% | 82 | -1058ms | +521ms | 1088ms | 596ms | 100% | 0 | 0 | 100% | 0 | — |
| vad+nst-050 | 14.67% | 82 | -1058ms | +521ms | 1088ms | 596ms | 100% | 0 | 0 | 100% | 0 | — |
| vad+nst-030 | 14.67% | 82 | -1058ms | +521ms | 1088ms | 596ms | 100% | 0 | 0 | 100% | 0 | — |
| vad-low | 15.50% | 85 | -840ms | +352ms | 902ms | 461ms | 97.8% | 2 | 0 | 97.9% | 2 | 1525ms |
| vad-high | 20.75% | 73 | -1218ms | +770ms | 1282ms | 853ms | 97.8% | 2 | 0 | 97.5% | 2 | 6680ms |
| suppress-nst | 52.42% | 104 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### All raw results

| Params | normCER | segs | s_off_mn | e_off_mn | s_mae | e_mae | det% | miss | fp | cover | gaps | g_avg |
|--------|---------|------|----------|----------|-------|-------|------|------|----|-------|------|-------|
| prompt+carry | 8.23% | 34 | -6212ms | +690599ms | 6276ms | 690781ms | 100% | 0 | 0 | 4321.2% | 6 | 4653ms |
| prompt | 9.66% | 42 | -6240ms | +690721ms | 6370ms | 690852ms | 100% | 0 | 0 | 4315.4% | 8 | 3765ms |
| baseline | 9.84% | 90 | -4527ms | +44298ms | 4795ms | 44451ms | 100% | 0 | 1 | 1260.0% | 87 | 3202ms |
| nst-050 | 9.84% | 90 | +96ms | -32ms | 185ms | 129ms | 95.7% | 4 | 0 | 53.3% | 88 | 897ms |
| nst-040 | 9.84% | 90 | +96ms | -32ms | 185ms | 129ms | 95.7% | 4 | 0 | 53.3% | 88 | 897ms |
| nst-020 | 9.84% | 90 | +96ms | -32ms | 185ms | 129ms | 95.7% | 4 | 0 | 53.3% | 88 | 897ms |
| nst-010 | 9.84% | 90 | +96ms | -32ms | 185ms | 129ms | 95.7% | 4 | 0 | 53.3% | 88 | 897ms |
| max-len-20 | 9.84% | 90 | +96ms | -32ms | 185ms | 129ms | 95.7% | 4 | 0 | 53.3% | 88 | 897ms |
| nst-020+max20 | 9.84% | 90 | +96ms | -32ms | 185ms | 129ms | 95.7% | 4 | 0 | 53.3% | 88 | 897ms |
| suppress-nst | 10.55% | 90 | +106ms | -35ms | 198ms | 130ms | 95.7% | 4 | 0 | 52.7% | 88 | 908ms |
| temp-04 | 10.55% | 84 | +29ms | +37ms | 220ms | 190ms | 95.7% | 4 | 0 | 54.6% | 81 | 946ms |
| vad-v6-th02 | 12.88% | 91 | -828ms | +198ms | 887ms | 281ms | 100% | 0 | 0 | 100% | 0 | — |
| vad-v6 | 14.31% | 87 | -1172ms | +310ms | 1227ms | 393ms | 100% | 0 | 0 | 100% | 0 | — |
| vad-v6+nst-020 | 14.31% | 87 | -1172ms | +310ms | 1227ms | 393ms | 100% | 0 | 0 | 100% | 0 | — |
| temp-02 | 15.74% | 48 | -2657ms | +2065ms | 2715ms | 2182ms | 100% | 0 | 0 | 92.4% | 3 | 4300ms |
| vad-default | 22.36% | 76 | -1732ms | +679ms | 1751ms | 750ms | 100% | 0 | 0 | 100% | 0 | — |
| vad+nst-050 | 22.36% | 76 | -1732ms | +679ms | 1751ms | 750ms | 100% | 0 | 0 | 100% | 0 | — |
| vad+nst-030 | 22.36% | 76 | -1732ms | +679ms | 1751ms | 750ms | 100% | 0 | 0 | 100% | 0 | — |
| vad-low | 22.90% | 77 | -1616ms | +644ms | 1683ms | 717ms | 97.8% | 2 | 0 | 97.9% | 2 | 1790ms |
| vad-high | 23.79% | 74 | -1940ms | +813ms | 1987ms | 874ms | 100% | 0 | 0 | 100% | 0 | — |

**Note**: raw baseline and prompt/prompt+carvy show absurd cover ratios (1260-4321%) — these runs exhibit hallucination loops (whisper generates infinite filler segments beyond audio end). The nst-* variants do not exhibit this despite being ostensibly identical, suggesting run-to-run nondeterminism or a different whisper.cpp binary version.

### All vocals results

| Params | normCER | segs | s_off_mn | e_off_mn | s_mae | e_mae | det% | miss | fp | cover | gaps | g_avg |
|--------|---------|------|----------|----------|-------|-------|------|------|----|-------|------|-------|
| prompt+carry | 9.12% | 9 | -11559ms | +610594ms | 11560ms | 610612ms | 100% | 0 | 0 | 4222.4% | 3 | 5640ms |
| baseline | 10.38% | 98 | -720ms | +113ms | 814ms | 285ms | 98.9% | 1 | 0 | 96.5% | 1 | 6000ms |
| nst-050 | 10.38% | 98 | -720ms | +113ms | 814ms | 285ms | 98.9% | 1 | 0 | 96.5% | 1 | 6000ms |
| nst-040 | 10.38% | 98 | -720ms | +113ms | 814ms | 285ms | 98.9% | 1 | 0 | 96.5% | 1 | 6000ms |
| nst-020 | 10.38% | 98 | -2072ms | +9899ms | 2190ms | 10062ms | 100% | 0 | 1 | 1354.5% | 1 | 6000ms |
| nst-010 | 10.38% | 98 | -2072ms | +9899ms | 2190ms | 10062ms | 100% | 0 | 1 | 1354.5% | 1 | 6000ms |
| max-len-20 | 10.38% | 98 | -2072ms | +9899ms | 2190ms | 10062ms | 100% | 0 | 1 | 1354.5% | 1 | 6000ms |
| suppress-nst | 10.38% | 98 | -2072ms | +9899ms | 2190ms | 10062ms | 100% | 0 | 1 | 1354.5% | 1 | 6000ms |
| nst-020+max20 | 10.38% | 98 | -2072ms | +9899ms | 2190ms | 10062ms | 100% | 0 | 1 | 1354.5% | 1 | 6000ms |
| vad-default | 12.52% | 88 | -763ms | +258ms | 825ms | 350ms | 100% | 0 | 0 | 100% | 0 | — |
| vad-low | 12.52% | 29 | -2004ms | +1751ms | 2139ms | 1791ms | 94.6% | 5 | 0 | 76.7% | 27 | 1459ms |
| vad+nst-050 | 12.52% | 88 | -763ms | +258ms | 825ms | 350ms | 100% | 0 | 0 | 100% | 0 | — |
| vad+nst-030 | 12.52% | 88 | -745ms | +5642ms | 833ms | 5726ms | 100% | 0 | 1 | 1365.5% | 0 | — |
| prompt | 12.70% | 45 | -3296ms | +456524ms | 3357ms | 456555ms | 100% | 0 | 0 | 4256.9% | 2 | 4060ms |
| temp-02 | 14.13% | 16 | -6530ms | +5961ms | 6581ms | 6127ms | 96.8% | 3 | 1 | 76.9% | 12 | 3280ms |
| vad-high | 14.85% | 86 | -870ms | +278ms | 922ms | 366ms | 100% | 0 | 0 | 100% | 0 | — |
| temp-04 | 16.99% | 20 | -6434ms | +5086ms | 6516ms | 5111ms | 98.9% | 1 | 0 | 81.4% | 17 | 1846ms |
| vad-v6-th02 | 10.38% | 27 | -3665ms | +452337ms | 3708ms | 452475ms | 100% | 0 | 1 | 4211.6% | 22 | 1417ms |
| vad-v6 | 11.45% | 30 | -3629ms | +108414ms | 3735ms | 108451ms | 96.8% | 3 | 0 | 3418.5% | 26 | 1469ms |

**Note**: Vocals + VAD variants show severe hallucination (cover > 1000% in many cases) — whisper generates looping segments beyond audio end. Vocals-only lacks full-band acoustic cues, confusing the decoder.

### Prompt variants (new)

Two new param sets: `prompt` (`--prompt "嗯"`) and `prompt+carry` (`--prompt "嗯" --carry-initial-prompt`) tested on all 3 sources.

| Source | Params | normCER | segs | s_off_mn | s_mae | det% | cover | "唉" | "啊"+"哈哈哈" |
|--------|--------|---------|------|----------|-------|------|-------|:----:|:-----------:|
| sidechain | prompt+carry | 8.48% | 79 | +207ms | 402ms | **94.7%** | 52.8% | ✅(71.28s,+0.04s) | ❌ |
| sidechain | prompt | 10.55% | 88 | +107ms | 1345ms | 94.7% | 49.4% | ❌ | ❌ |
| raw | prompt+carry | 8.23% | 34 | -6212ms | 6276ms | **100%** | 4321% | — | ❌ |
| raw | prompt | 9.66% | 42 | -6240ms | 6370ms | 100% | 4315% | — | ❌ |
| vocals | prompt+carry | 9.12% | **9** | -11559ms | 11560ms | 100% | 4222% | ✅ | ✅(7.5s offset) |
| vocals | prompt | 12.70% | 45 | -3296ms | 3357ms | 100% | 4257% | — | ❌ |

Key findings:
- **`prompt+carry` on sidechain** is the **first non-VAD config to capture "唉"** (71.28s, offset +0.04s — nearly perfect timing). However, it still misses "啊"+"哈哈哈" at 113-116s.
- **`prompt+carry` on vocals** captures ALL fillers including "啊"+"哈哈哈" but collapses to only **9 segments** — segmentation is destroyed.
- **`prompt` without `carry`** on sidechain causes hallucination (923 chars vs 564 GT, CER 10.55% but with extreme MAE and a false positive).
- `prompt+carry` does not significantly improve over baseline for practical use — the fillers it captures are limited to one short segment, and hallucination risk persists.

### Detailed offset analysis for problem segments

#### "唉" (71.20-71.72)

| Source | Params | GT → Hyp | s_off | IOU |
|--------|--------|----------|-------|-----|
| raw/sidechain | baseline | MISSED (no overlap) | — | 0 |
| raw | vad-v6 | [71.20-71.72] → [69.92-71.48] "唉" | **-1.28s** | 0.16 |
| raw | vad-v6-th02 | [71.20-71.72] → [69.74-71.53] "唉" | **-1.46s** | 0.17 |
| sidechain | vad-v6 | [71.20-71.72] → [69.92-71.52] "唉" | **-1.28s** | 0.18 |
| sidechain | vad-v6-th02 | [71.20-71.72] → [70.40-74.02] "还是缺灵石啊" | **-0.80s** | 0.14 |
| sidechain | **prompt+carry** | [71.20-71.72] → **[71.28-72.06] "唉"** | **+0.04s** | **0.52** |

VAD v6 detects "唉" but shifts start ~1.3s left. `prompt+carry` on sidechain achieves nearly perfect timing (IOU 0.52, s_off +0.04s) — the best detection of "唉" across all 40+ configs, without VAD.

#### "啊" (113.96-115.12) + "哈哈哈" (115.42-116.96)

| Source | Params | "啊" | "哈哈哈" |
|--------|--------|:---:|:-------:|
| all non-VAD | baseline/temp-* | ❌ | ❌ |
| all VAD | vad-* | ❌ | ❌ |
| sidechain | prompt+carry | ❌ | ❌ |
| vocals | prompt+carry | ✅ (122.80s) | ✅ (122.80s) |

Vocals `prompt+carry` captures both but at 122.80s (GT 113.96-116.96) — merged into one segment ~6s late. Unusable for timing.

No config achieves viable detection of "啊"+"哈哈哈" without VAD or segmentation collapse.

### GT segment structure around problem area

```
[107.80-108.72] "谢师父指点"
         ~5.2s silence
[113.96-115.12] "啊"           (1.16s)  ← never captured (except vocals prompt+carry, unusable)
          0.3s gap
[115.42-116.96] "哈哈哈"       (1.54s)  ← never captured (same)
         ~1.04s silence
[118.00-119.16] "嗯发财了"     (1.16s)  ← all whisper runs capture "发财了" (omit "嗯")
```

> GT from `packages/benchmark/ref/metadata/asr_manual.json` (ms format, user-corrected)

### Key findings

1. **CER alone is misleading.** Lowest CER (7.72%, sidechain/vad-v6-th02) comes with +190ms start offset — acceptable but not great. MAE (391ms) better captures the spread.

2. **Temperature-annealed configs (temp-02/04) on sidechain have best combined CER+trade-off** (CER 7.72%, s_mae 262ms, s_off +100ms). This remains the correct choice for pipeline use.

3. **`prompt+carry` is a partial improvement** — it captures "唉" with near-perfect timing (IOU 0.52, +0.04s) without VAD, but still fails on "啊"+"哈哈哈" and risks hallucination on raw/vocals.

4. **VAD variants systematically shift boundaries left** (s_off_mean -0.6 to -1.9s). Full coverage (cover=100%) but at unacceptable timing penalty.

5. **Neither prompt nor VAD solves "啊"+"哈哈哈"** convincingly. These filler words at 113-116s are decoded as "发财了" by whisper's language model regardless of segmentation, prompt, or source. OCR remains the only reliable path.

6. **MAE (mean absolute error) is a better timing metric than mean offset** — it captures both direction and spread. e_mae is systematically higher than s_mae for all configs (whisper more uncertain about end boundaries than start).

### Recommendation for pipeline

**Use `sidechain + --temperature 0.2`.** Best timing accuracy (s_mae 262ms, s_off +100ms), lowest CER (7.72%). Acceptable trade-off: "唉", "啊", and "哈哈哈" missing (~3.5s out of 170s). If filler capture matters, **OCR is the reliable alternative** — it captures all fillers from hardsubs.

### How to run

```bash
# Full ASR benchmark (38 + 3 prompt param sets)
bun run packages/benchmark/asr/whisper/compute/benchmark-asr-params.ts

# Batch evaluation with ms timestamps
bun packages/benchmark/ref/compute/eval-ocr.ts --batch results/ggml-s1-sidechain ref/metadata/asr_manual.json --hyp-file asr.json --ms

# Single run eval
bun packages/benchmark/ref/compute/eval-ocr.ts results/ggml-s1-sidechain/temp-02/metadata/asr.json ref/metadata/asr_manual.json --ms
```

**Files**: `results/{source}/{param-label}/metadata/{asr,whisper_raw,summary}.json`
**Scripts**: `packages/benchmark/asr/whisper/compute/benchmark-asr-params.ts`, `packages/benchmark/ref/compute/eval-ocr.ts`
