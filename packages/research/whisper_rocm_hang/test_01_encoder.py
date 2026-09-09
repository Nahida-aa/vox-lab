#!/usr/bin/env python3
"""
test_01_encoder.py — Quantify GPU Hang intermittency for Whisper encoder.

Measures success/hang rate across trials for different input shapes:
  - Random mel (3000 frames, padded)
  - Real audio mel (variable length, from a WAV file)
  - Small vs large batch sizes

Usage:
    cd YouDub-webui && .venv/bin/python packages/research/whisper-rocm-hang/test_01_encoder.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_rocm_hang.setup import run_trials, summary_table, device_info


RANDOM_ENCODER_CODE = r"""
import os, sys
sys.path.insert(0, '.')

from whisper_rocm_hang.setup import _enable_conv_patch, load_whisper_model
_enable_conv_patch()

import torch

model = load_whisper_model("large-v3-turbo", device="cuda")

for shape in [
    (1, 128, 3000),    # standard padded
    (1, 128, 1906),    # ~19s audio (variable)
    (4, 128, 3000),    # batch=4
]:
    mel = torch.randn(*shape, device="cuda")
    with torch.no_grad():
        out = model.encoder(mel)
    print(f"shape={shape} OK {list(out.shape)}")
"""

REAL_AUDIO_ENCODER_CODE = r"""
import os, sys
sys.path.insert(0, '.')

from whisper_rocm_hang.setup import _enable_conv_patch, load_whisper_model
from pathlib import Path
_enable_conv_patch()

import torch
import whisper.audio as wa

model = load_whisper_model("large-v3-turbo", device="cuda")

# Use the Me at the zoo test audio
vocals = Path("workfolder/jawed/Me_at_the_zoo__jNQXAC9IVRw/media/audio_vocals.wav")
if vocals.exists():
    audio = wa.load_audio(str(vocals))
    mel = wa.log_mel_spectrogram(audio, n_mels=128)
    mel = mel.unsqueeze(0).to("cuda")
    with torch.no_grad():
        out = model.encoder(mel)
    print(f"real audio OK {list(out.shape)}")
else:
    print(f"real audio NOT FOUND, skip")
"""


def main():
    print("=== test_01_encoder: GPU Hang intermittency ===")
    print(f"Device info: {device_info()}")
    print()

    print("--- Test A: Random mel, 3000 frames ---")
    results_a = run_trials(RANDOM_ENCODER_CODE, trials=5, timeout=90,
                           label="encoder(random mel 3000)")
    print()
    print(summary_table(results_a, "encoder(random mel 3000)"))
    print()

    print("--- Test B: Real audio ---")
    results_b = run_trials(REAL_AUDIO_ENCODER_CODE, trials=5, timeout=90,
                           label="encoder(real audio)")
    print()
    print(summary_table(results_b, "encoder(real audio)"))


if __name__ == "__main__":
    main()
