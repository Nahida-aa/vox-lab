#!/usr/bin/env python3
"""
test_04_faster_whisper.py — Test faster-whisper (CTranslate2) on GPU + CPU.

faster-whisper uses CTranslate2 inference engine, not PyTorch ops.
If the ROCm hang is PyTorch-specific, faster-whisper may work on GPU.

Results so far (exploratory):
  - GPU (cuda): ❌ "CUDA driver version is insufficient" — prebuilt
    ctranslate2 wheel requires real CUDA driver, ROCm HIP shim not compatible
  - CPU (int8): ✅ RTF ~0.25, accurate, word_timestamps work

Usage:
    .venv/bin/python packages/research/whisper_rocm_hang/test_04_faster_whisper.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whisper_rocm_hang.setup import run_trials, summary_table, device_info


FASTER_GPU_CODE = r"""
import os, sys
sys.path.insert(0, '.')
from faster_whisper import WhisperModel

model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
segments, info = model.transcribe(
    'workfolder/jawed/Me_at_the_zoo__jNQXAC9IVRw/media/audio_vocals.wav',
    language='en',
)
text = ' '.join(s.text for s in segments)
print(f"OK ({info.duration:.1f}s audio): {text[:100]}")
"""

FASTER_CPU_CODE = r"""
import os, sys
sys.path.insert(0, '.')
from faster_whisper import WhisperModel

model = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')
segments, info = model.transcribe(
    'workfolder/jawed/Me_at_the_zoo__jNQXAC9IVRw/media/audio_vocals.wav',
    language='en',
)
text = ' '.join(s.text for s in segments)
print(f"OK ({info.duration:.1f}s audio): {text[:100]}")
"""


def main():
    print("=== test_04_faster_whisper: CTranslate2 engine ===")
    print(f"Device info: {device_info()}")
    print()

    print("--- Test A: faster-whisper GPU (cuda) ---")
    results_gpu = run_trials(FASTER_GPU_CODE, trials=3, timeout=120,
                             label="faster-whisper GPU")
    print()
    print(summary_table(results_gpu, "faster-whisper GPU"))
    print()

    print("--- Test B: faster-whisper CPU (int8) ---")
    results_cpu = run_trials(FASTER_CPU_CODE, trials=3, timeout=120,
                             label="faster-whisper CPU")
    print()
    print(summary_table(results_cpu, "faster-whisper CPU"))


if __name__ == "__main__":
    main()
