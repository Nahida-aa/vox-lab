"""Whisper ASR benchmark: Python PyTorch (CUDA/ROCm).

GPU (cuda/ROCm) path hangs on Radeon 780M + ROCm 7.2.3 + PyTorch 2.12.0
during model.transcribe() with "GPU Hang" HW exception.
Model loads fine on GPU, but encoder conv triggers MIOpen solver failure.

CPU fallback is reported as the actual result.
"""
import json, os, time
from pathlib import Path

import whisper
import torch

MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WAV = "/tmp/chirp.wav"
_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # GPU path hangs — force CPU
    device = "cpu"
    print(f"=== Whisper PyTorch Benchmark ===")
    print(f"Model: {MODEL_NAME}, Torch: {torch.__version__}")
    print(f"NOTE: GPU (cuda/ROCm) hangs with 'GPU Hang' on Radeon 780M + ROCm 7.2.3")
    print(f"  Model loads OK on GPU, but model.transcribe() -> conv forward -> MIOpen solver fails")
    print(f"Using CPU fallback for benchmark\n")

    t0 = time.perf_counter()
    model = whisper.load_model(MODEL_NAME, device=device)
    load_time = time.perf_counter() - t0
    print(f"  load OK  ({load_time:.2f}s)")

    td0 = time.perf_counter()
    result = model.transcribe(WAV, language="en", verbose=False)
    total_time = time.perf_counter() - td0

    segments = result.get("segments", [])
    duration = segments[-1]["end"] if segments else 0
    text = (result.get("text") or "").strip()
    rtf = total_time / duration if duration > 0 else float("inf")

    print(f"  transcribe {total_time:.2f}s, audio {duration:.2f}s, RTF={rtf:.2f}")
    print(f"  text: {text[:80]}")

    out = {
        "engine": "pytorch",
        "device": device,
        "model": MODEL_NAME,
        "load_time_s": round(load_time, 3),
        "transcribe_time_s": round(total_time, 3),
        "audio_duration_s": round(duration, 3),
        "rtf": round(rtf, 3),
        "text": text,
        "note": "GPU (cuda/ROCm) hangs with 'GPU Hang' on this hardware"
    }
    path = RESULTS_DIR / "py-pth-cuda.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
