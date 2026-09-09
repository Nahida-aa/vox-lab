#!/usr/bin/env python3
"""
test_02_transcribe.py — Compare GPU Hang across model sizes (tiny vs large-v3-turbo).

Determines if the hang is model-size dependent:
  - tiny (39M params) — minimal memory pressure
  - large-v3-turbo (~2B params) — full model

Also tests:
  - with/without conv_patch
  - with/without word_timestamps
  - with/without without_timestamps decode option

Usage:
    .venv/bin/python packages/research/whisper-rocm-hang/test_02_transcribe.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_rocm_hang.setup import run_trials, summary_table, device_info

TINY_TRANSCRIBE_CODE = r"""
import os, sys
sys.path.insert(0, '.')

from whisper_rocm_hang.setup import load_whisper_model
from pathlib import Path

model = load_whisper_model("tiny", device="cuda")

vocals = Path("workfolder/jawed/Me_at_the_zoo__jNQXAC9IVRw/media/audio_vocals.wav")
if vocals.exists():
    result = model.transcribe(str(vocals), language="en", verbose=False)
    text = (result.get("text") or "").strip()
    print(f"tiny OK ({len(text)} chars): {text[:100]}")
else:
    print("audio NOT FOUND")
"""

LARGE_TRANSCRIBE_CODE = r"""
import os, sys
sys.path.insert(0, '.')

from whisper_rocm_hang.setup import _enable_conv_patch, load_whisper_model
from pathlib import Path

_enable_conv_patch()

model = load_whisper_model("large-v3-turbo", device="cuda")

vocals = Path("workfolder/jawed/Me_at_the_zoo__jNQXAC9IVRw/media/audio_vocals.wav")
if vocals.exists():
    result = model.transcribe(str(vocals), language="en", verbose=False)
    text = (result.get("text") or "").strip()
    print(f"large OK ({len(text)} chars): {text[:100]}")
else:
    print("audio NOT FOUND")
"""

LARGE_NO_TIMESTAMPS_CODE = r"""
import os, sys
sys.path.insert(0, '.')

from whisper_rocm_hang.setup import _enable_conv_patch, load_whisper_model
from pathlib import Path

_enable_conv_patch()

model = load_whisper_model("large-v3-turbo", device="cuda")

vocals = Path("workfolder/jawed/Me_at_the_zoo__jNQXAC9IVRw/media/audio_vocals.wav")
if vocals.exists():
    result = model.transcribe(str(vocals), language="en", verbose=False, word_timestamps=False)
    text = (result.get("text") or "").strip()
    print(f"large(no_ts) OK ({len(text)} chars): {text[:100]}")
else:
    print("audio NOT FOUND")
"""


def main():
    print("=== test_02_transcribe: Model size comparison ===")
    print(f"Device info: {device_info()}")
    print()

    print("--- Test A: tiny (39M) full transcribe ---")
    results_a = run_trials(TINY_TRANSCRIBE_CODE, trials=5, timeout=120,
                           label="tiny transcribe")
    print()
    print(summary_table(results_a, "tiny transcribe"))
    print()

    print("--- Test B: large-v3-turbo (~2B) + conv_patch + word_timestamps ---")
    results_b = run_trials(LARGE_TRANSCRIBE_CODE, trials=5, timeout=180,
                           label="large transcribe")
    print()
    print(summary_table(results_b, "large transcribe"))
    print()

    print("--- Test C: large-v3-turbo + conv_patch, no word_timestamps ---")
    results_c = run_trials(LARGE_NO_TIMESTAMPS_CODE, trials=5, timeout=180,
                           label="large transcribe (no word_timestamps)")
    print()
    print(summary_table(results_c, "large transcribe (no word_timestamps)"))


if __name__ == "__main__":
    main()
