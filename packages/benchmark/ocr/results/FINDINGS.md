# OCR Benchmark Results

Production config: **fps=2, subtitle-only, text_score=0.45** (342 frames from 170s 720p anime video).

## 5-Engine Comparison (fps2-so-ts0.45)

| Engine                | Frames | Segs | CER       | FP  | Missed | Inf (s)  | RTF       | vs C++          |
| --------------------- | ------ | ---- | --------- | --- | ------ | -------- | --------- | --------------- |
| **C++ (PP-OCRv3)**    | 341    | 75   | **0.36%** | 0   | 0      | 91.6     | 0.539     | —               |
| Python (PP-OCRv3)     | 341    | —    | —         | —   | —      | ~180     | ~1.06     | —               |
| Node.js (PP-OCRv3)    | 341    | —    | —         | —   | —      | —        | —         | —               |
| **oar-ocr (v6 tiny)** | 341    | 88   | **6.44%** | 7   | 0      | **30.8** | **0.181** | **3.0× faster** |
| Rust ORT (PP-OCRv3)   | 341    | —    | —         | —   | —      | —        | —         | —               |

C++ at fps2 achieves **near-perfect accuracy**: 0.36% CER (2 chars wrong, normalized to 1), 0 missed segments, 0 false positives, exact match on all 75 GT segments.

oar-ocr is **3× faster** (RTF 0.181 vs 0.539, ~90ms/frame) but trades accuracy for speed:

- **CER 6.44%** (6ppt higher) — mainly from 7 false positive segments ("Ra", "E", "11", "11AA1", "14144449", "OTTO", "mE") from scene text/UI digits leaking through.
- **0 missed GT segments** — detection coverage matches C++ at fps2.
- False positives are all scene noise (bottom-of-frame UI text) that don't get filtered because PP-OCRv6 tiny's detection + oar-ocr's `use_dilation=false` produces weaker contours.

## Root Cause

PP-OCRv3 models are incompatible with oar-ocr v0.7.1 — `use_dilation=false` is hardcoded in `TextDetectionAdapterBuilder` (C++/Python use `true`). Without dilation, the thresholded heatmap mask is too thin/fragmented for contour detection. PP-OCRv6 tiny detects text but with lower precision → scene noise leaks through.

## Verdict

**oar-ocr not recommended for production.** Speed advantage (3× vs C++) does not justify 6ppt higher CER. Would be viable if:

- `use_dilation` is exposed as config knob → enables PP-OCRv3 support
- Or PP-OCRv4 detection model tested with better tiny-model accuracy

## Key Files

- `packages/benchmark/ocr/oar-rs/` — Rust binary wrapping oar-ocr v0.7.1 with PP-OCRv6 tiny
- `data/models/rapidocr/` — Universal OCR model path (PP-OCRv3 + v6 tiny ONNX models)
- `packages/benchmark/ocr/compute/benchmark-ocr-video.ts` — Orchestrator, supports `--engine oar`
- `packages/subtitle-ocr/ort-cpp/ocr_pipeline.cpp` — C++ ORT pipeline (production reference)
