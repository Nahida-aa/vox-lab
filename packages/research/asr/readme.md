# ASR

## Active paths

| 模型 | 设备 | 说明 |
|------|------|------|
| whisper.cpp | Vulkan (RADV) | RTF ~0.09-0.11 (参数相关)，large-v3-turbo |

## Failed paths

- whisper-pytorch GPU: segfault
- faster-whisper GPU: 缺 libcuda.so
- whisper.cpp HIPBLAS: MES hang
- VoxCPM: 所有路径均已废弃

详细原因 → `.agents/hardware.md` / `.agents/model-strategy.md`


## ASR evaluation

### Script

`packages/benchmark/ref/compute/eval-ocr.ts` — unified ASR evaluation with text normalization.

Usage:
```
bun eval-ocr.ts <hyp.json> [gt.json] [--label <label>]
bun eval-ocr.ts --batch <results_dir> [gt.json]
```

Normalization rules:
1. `师父` → `师傅` (homophone unification)
2. Strip whitespace & punctuation
3. Numerals (Arabic + Chinese) → `#` placeholder

### Caveat on timestamp offset

Offset (e.g. start +200ms, end -130ms) is **not necessarily a timing error**. ASR systematically skips filler words (`啊`, `嗯`, `哎`, `哈哈哈`, `啊啊`) and low-volume speech at segment boundaries. This causes:
- Positive start offset: ASR starts later because it dropped leading filler
- Negative end offset: ASR ends earlier because it dropped trailing filler

These offsets appear even when the ASR timing is perfect. They should not be penalized as timing errors unless the content difference confirms misalignment. Always check `missed` segments — if they are filler-only, the offset is likely justified.

### Cross-source findings (sidechain vs raw vs vocals)

Sidechain processing is **critical** for `temp-*` parameters — without it, segment count collapses and timing becomes unusable:

| Params | Audio | normCER | Segs | s_off | Det. | Cover | Gaps | Gap avg |
|--------|-------|---------|------|-------|------|-------|------|---------|
| vad-v6-th02 | sidechain | **8.41%** | 88 | -534ms | 100% | 95.7% | 2 | 3.7s |
| vad-v6-th02 | raw | 13.06% | 91 | -737ms | 100% | 100% | 0 | — |
| vad-v6 | sidechain | 12.52% | 88 | -584ms | 100% | 100% | 0 | — |
| vad-v6 | raw | 14.49% | 87 | -884ms | 100% | 100% | 0 | — |
| vad-default | sidechain | 14.67% | 82 | -849ms | 100% | 100% | 0 | — |
| vad-default | raw | 22.36% | 76 | -1145ms | 100% | 100% | 0 | — |
| temp-02 | sidechain | **9.48%** | 86 | +203ms | 94.7% | 54.4% | 83 | 0.93s |
| temp-02 | raw | **15.92%** | **48** | -2342ms | 100% | 93.0% | 46 | 4.3s |
| temp-02 | vocals | **14.31%** | **16** | -6357ms | 94.7% | 77.3% | 15 | 3.3s |
| temp-04 | sidechain | 9.48% | 86 | +204ms | 94.7% | 54.4% | 83 | 0.93s |
| temp-04 | raw | 10.55% | 84 | +175ms | 94.7% | 54.6% | 80 | 0.95s |
| temp-04 | vocals | **16.99%** | **20** | -5887ms | 98.7% | 81.3% | 18 | 1.8s |

Key: `temp-*` params are extremely sensitive to BGM interference; sidechain is required. `vad-v6-th02` degrades less on raw (+4.65ppt) but still benefits from sidechain. Vocals-only makes timing worse for all params due to reduced acoustic cues at segment boundaries.
