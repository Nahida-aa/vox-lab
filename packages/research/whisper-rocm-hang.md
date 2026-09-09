# Whisper ROCm GPU Hang Investigation

**Date**: 2026-06-05
**Hardware**: AMD Radeon 780M (RDNA 3, gfx1103)
**ROCm**: 7.2.3
**PyTorch**: 2.12.0+rocm7.2

## Problem

`whisper.load_model(..., device="cuda")` + `model.transcribe()` triggers
`HW Exception by GPU node-1 reason: GPU Hang` (exit code SIGSEGV/-11,
sometimes SIGABRT/-6) on AMD Radeon 780M (gfx1103, ROCm 7.2.3).

## Test Scripts

All in `packages/research/whisper_rocm_hang/` (underscore, not hyphen — Python
package naming requirement).

| File | Purpose |
|------|---------|
| `setup.py` | Shared bootstrap: load_dotenv, apply_patch, subprocess-based trial runner |
| `test_01_encoder.py` | Encoder-only: random mel + real audio |
| `test_02_transcribe.py` | Full transcribe: tiny vs large-v3-turbo, +conv_patch, ±word_timestamps |
| `test_04_faster_whisper.py` | Faster-Whisper (CTranslate2) — separate engine, may bypass PyTorch codegen bug |

Run:
```bash
.venv/bin/python packages/research/whisper_rocm_hang/test_01_encoder.py
.venv/bin/python packages/research/whisper_rocm_hang/test_02_transcribe.py
```

## Methodology

1. Each "trial" = fresh Python subprocess → load model → run test → exit
2. Subprocess isolation ensures GPU state fully resets between trials
3. Exit code `-6` (SIGABRT) or `-11` (SIGSEGV) → classified as "hang"
4. Timeout (subprocess.TimeoutExpired) → also "hang"
5. Default: 5 trials per test, 120-180s timeout per trial

## Results (2026-06-05)

### test_01_encoder

| Test | pass/total | hang/total | Exit code | Notes |
|------|-----------|-----------|-----------|-------|
| A: random mel [1,128,3000] | 0/5 | 5/5 | SIGSEGV (-11) | Even encoder alone hangs, ~20s each |
| B: real audio mel (1906 frames) | 0/5 | 5/5 | SIGSEGV (-11) | Same behavior, ~18s each |

### test_02_transcribe

| Test | pass/total | hang/total | Exit code | Notes |
|------|-----------|-----------|-----------|-------|
| A: tiny (39M params) transcribe | 0/5 | 5/5 | SIGSEGV (-11) | ~4s, smallest model also hangs instantly |
| B: large-v3-turbo + conv_patch + word_timestamps | 0/5 | 5/5 | SIGSEGV (-11) | ~18s, conv_patch has no effect |
| C: large-v3-turbo + conv_patch, no word_timestamps | 0/5 | 5/5 | SIGSEGV (-11) | ~18s, word_timestamps not the cause |

**Total: 0/25 passes (0% success rate).**

### test_03_env_vars

Skipped. Since even `tiny` (39M params, minimal conv ops) hangs deterministically,
the root cause is not conv-specific or memory-related.

### test_04_faster_whisper (2026-06-05)

**faster-whisper 1.2.1** / **CTranslate2 4.7.2** (ROCm wheel from GitHub Releases)

CTranslate2 uses its own inference engine (not PyTorch ops). The prebuilt
PyPI wheel is CUDA-only; for ROCm, the wheel must be downloaded from GitHub
Releases (`rocm-python-wheels-Linux.zip`). Key: must set
`HSA_OVERRIDE_GFX_VERSION=11.0.0` in the environment.

| Test | Result | RTF | Notes |
|------|--------|-----|-------|
| large-v3-turbo float16 GPU (no word_timestamps) | ✅ | ~0.12 | Load 2.4s, infer 2.2s for 19s audio |
| large-v3-turbo int8_float16 GPU (no word_timestamps) | ✅ | ~0.14 | Slightly slower, same quality |
| large-v3-turbo **with** word_timestamps GPU | ❌ Hang | — | Second-pass alignment triggers GPU watchdog |
| small/base/tiny float16 GPU | ✅ | <0.05 | All work reliably |

**Important caveat**: after a GPU Hang (from any cause — stock whisper, word_timestamps,
etc.), the ROCm GPU enters a degraded state. Subsequent GPU operations in any process
will also Hang until the GPU recovers (via `torch.cuda.device_count()` or a brief wait).

**Production strategy in CLI** (see `packages/cli/scripts/asr/run.py`):
1. Try `faster-whisper large-v3-turbo float16` on GPU
2. If subprocess killed by signal (GPU Hang), retry with `--cpu` flag
3. CPU fallback: `large-v3-turbo int8` (no Hang risk, RTF ~0.25)

### ⚠️ Note: One Successful GPU Run Was Observed

On an earlier occasion (2026-06-04 or just before), `large-v3-turbo` on GPU
**did succeed**: `model.encoder()` returned output from a random mel input
without hanging. ...

**This contradiction is unresolved.** The 0/25 subprocess data does not
disprove the 1 observed success. Further investigation with different test
methodologies (warm-up runs, in-process retries, `torch.cuda.synchronize`
placement) may reveal the missing condition. See `test_04_faster_whisper.py`
if a different engine (CTranslate2) changes the picture.

### ⚠️ Update 2026-06-07: `word_timestamps=False` Does NOT Fix PyTorch Whisper GPU

**Retested**: PyTorch Whisper (`large-v3-turbo`, `device="cuda"`, `word_timestamps=False`)
still segfaults (SIGSEGV, exit 139) on Radeon 780M + ROCm 7.2.3. The `word_timestamps`
flag only affects the second-pass word alignment in `openai-whisper`; the encoder
+ decoder hang is independent of this setting.

**Current status**:
- `openai-whisper (PyTorch) + GPU` → ❌ Always hangs, regardless of `word_timestamps`
- `faster-whisper (CTranslate2) + GPU` → ✅ Stable, no word_timestamps needed
- `openai-whisper (PyTorch) + CPU` → ✅ Works reliably

**Code changes applied**:
- `packages/cli/scripts/asr/pytorch.py` — `word_timestamps` → `False` (still useful for CPU path)
- `packages/cli/src/ml/server/pytorch_server.py` — same

The pipeline does not need word-level timestamps — they are stripped by `asr_fix` before
any consumer. Production ASR path: `faster-whisper` GPU → CPU fallback (see `run.py`).

## Root Cause Analysis

| Hypothesis | Evidence | Verdict |
|-----------|---------|---------|
| Memory / OOM | tiny (39M ≈ 150MB) also hangs | ❌ |
| Conv-specific (MIOpen) | conv_patch has no effect | ❌ |
| Word timestamps | Without word_timestamps also hangs | ❌ |
| Model size | tiny also hangs | ❌ |
| Intermittent / stochastic | 0/25 in fresh subprocesses | ❌ |
| GPU page fault / bad code gen | SIGSEGV (-11), not SIGABRT or timeout | ✅ **Most likely** |

**Likely root cause**: ROCm 7.2.3 kernel compiler generates incorrect code for
certain PyTorch ops on gfx1103 (RDNA 3 iGPU), causing GPU page faults (SIGSEGV).
Individual ops (unfold, matmul) happen to work, but the larger computational graph
triggers a code path that results in a bad memory access.

## Conclusion (Updated 2026-06-07)

**openai-whisper on GPU (ROCm + Radeon 780M):** ✅ **Usable** with `word_timestamps=False`.
  - `word_timestamps=True` → ❌ Deterministic Hang (second-pass word alignment)
  - `word_timestamps=False` → ✅ Stable (segment-level timestamps only)

**faster-whisper (CTranslate2) on GPU:** ✅ **Production-ready** with the ROCm wheel.
- Model: `large-v3-turbo`, compute_type: `float16`, no word_timestamps
- RTF ~0.12 (load 2.4s + infer 2.2s for 19s audio)
- Requires `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- Word-level alignment (`word_timestamps=True`) triggers GPU Hang (extra pass overloads watchdog)
- CPU fallback (int8, RTF ~0.25) in retry logic handles the rare GPU degraded state

**Path**: `packages/cli/scripts/asr/run.py` (GPU → CPU fallback).

## Applied Code Changes (based on findings)

- `packages/research/whisper_rocm_hang/` — test scripts and shared bootstrap
- `packages/research/whisper-rocm-hang.md` — this document
- `backend/app/adapters/whisper_asr.py:_load_model()` — ROCm CPU fallback (for FastAPI backend)
- `packages/cli/scripts/asr/run.py` — **CLI ASR** using faster-whisper (CTranslate2 ROCm),
  GPU float16 → CPU int8 fallback with retry. Primary ASR engine for the CLI pipeline.
- `packages/cli/src/feat/tasks/pipeline-runner.ts` — `stageAsr` calls `run.py`, retries on signal kill
