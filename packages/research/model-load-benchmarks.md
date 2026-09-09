# Model Load Benchmarks & OOM Analysis

**Date**: 2026-06-06/07
**Hardware**: AMD Radeon 780M (RDNA 3, gfx1103), 27GB RAM + 27GB zram swap, ROCm 7.2.3
**Python**: 3.12 (`.venv/bin/python`), PyTorch 2.12.0+rocm7.2
**Test video**: GitHub asset (~10s, source: `bd02936f-cf3c-4e4b-85b5-0410d38f69f5`)
**Pipeline**: dub mode, 9 stages, all models on CPU unless noted

## Load Times

Measured from `task_stages.started_at` / `completed_at` in SQLite.

| Model | File Size | Load Time | Inference Time | Total | Runtime | Device |
|-------|-----------|-----------|----------------|-------|---------|--------|
| Demucs (htdemucs_ft) | ~400MB | ~40-50s | ~246s | 296s | PyTorch | CPU |
| faster-whisper (large-v3-turbo) | ~1.5GB | ~3-5s | ~7s | 12s | CTranslate2 | GPU |
| Whisper (openai-whisper) | ~3GB | ~15-30s | ~30s+ | N/A | PyTorch | CPU/GPU |
| VoxCPM2 (1.3B) | ~4.7GB | ~35s | ~462s (7 clips) | 497s | PyTorch | CPU |

Load time is the delta between stage start and first progress/complete message, excluding inference.

### Notes

- **Demucs loads slowly** despite being the smallest model because `torch.hub.load()` + `Separator(...)` initialization includes module import, PyTorch model instantiation, and weight loading.
- **faster-whisper (CTranslate2)** is the fastest loader — its quantized binary format is memory-mapped and nearly instant.
- **VoxCPM `from_pretrained`** loads `model.safetensors` (4.3GB, mmap), `audiovae.pth` (360MB), creates model architecture, and by default runs a warmup `generate()` with `max_len=10`. The warmup adds ~5-10s to load time.

## Peak Memory (RSS)

All measurements from kernel OOM reports + `free -h`.

| Configuration | Peak RSS | Result |
|--------------|----------|--------|
| Demucs (CPU) + VoxCPM (CPU) — torch server | ~5.5GB | ✅ Works |
| Demucs (CPU) + Whisper PyTorch (CPU) + VoxCPM (CPU) — torch server | ~9.5GB | ❌ OOM killed |
| Demucs (CPU) + VoxCPM (CPU) — standalone subprocess per stage | ~4-5GB | ✅ Works |
| Any single model standalone | ~2-5GB | ✅ Works |

### OOM Boundary

```
27GB RAM + 27GB swap
System baseline (KDE, browser, etc.): ~17GB RAM, ~23GB swap used
Available headroom: ~10GB RAM, ~4GB swap
```

- **Whisper PyTorch on GPU**: segfaults (likely ROCm regression, was stable before)
- **Whisper PyTorch on CPU**: ~3GB RSS added to torch server process
- **Three models concurrently in torch server**: exceeds 10GB headroom → OOM
- **Two models (Demucs + VoxCPM) in torch server**: fits within limits
- **faster-whisper (subprocess)**: loaded in a separate process that exits after ASR, so its memory is reclaimed before VoxCPM starts

## Device Compatibility

| Model | Runtime | GPU (cuda) | CPU |
|-------|---------|------------|-----|
| Demucs | PyTorch | ❌ GPU hang (htdemucs, shifts=3) | ✅ RTF ~2.0 |
| Whisper | PyTorch (openai-whisper) | ✅ (word_timestamps=False) | ✅ |
| Whisper | CTranslate2 (faster-whisper) | ✅ ("cuda" via CTranslate2) | ✅ |
| VoxCPM | PyTorch (from_pretrained) | ❌ GPU segfault | ✅ (~4.7GB weights) |
| VoxCPM | ONNX (onnxruntime) | ⏸️ Dawn WebGPU leak / MIGraphX 10x slower | ✅ sequential load, peak ~16GB |

## Torch Server Architecture Takeaways

The TCP torch server works correctly when model memory fits:

- **Model caching is viable** when only 1-2 moderate models (<6GB total) share the torch server process
- **faster-whisper as subprocess**: ideal — GPU-accelerated, exits after use, no torch server memory impact
- **VoxCPM on CPU**: load + generate within torch server works (35s + 462s for 7 clips)
- **Demucs on CPU**: works but slow (296s for ~2min audio)

If all three models must be CPU, use standalone subprocesses (torch server not started) to avoid OOM.

## VoxCPM TTS Performance (CPU)

```
7 clips generated in 462.3s
Average: 66s per clip (~10s audio per clip → RTF ~6.6)
Total TTS stage: 497s (35s load + 462s generate + overhead)
```

RTF is in the expected range for CPU VoxCPM (AGENTS.md: RTF ~7.4 for ONNX CPU).

## References

- AGENTS.md — device assignment strategy
- `.agents/model-strategy.md` — model-to-device mapping rationale
- `packages/research/voxcpm-onnx-nan.md` — VoxCPM ONNX NaN root cause
- `packages/research/whisper-rocm-hang.md` — Whisper GPU hang on ROCm
- `docs/webgpu-oom.md` — Dawn WebGPU `VK_ERROR_DEVICE_LOST`
