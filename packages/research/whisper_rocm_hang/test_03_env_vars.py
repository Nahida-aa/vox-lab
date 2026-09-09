#!/usr/bin/env python3
"""
test_03_env_vars.py — Test environment variable combinations that may
work around the Whisper ROCm GPU Hang.

Variables tested:
  - HSA_ENABLE_SDMA=0       (disable SDMA, force sync DMA)
  - HSA_OVERRIDE_GFX_VERSION=11.0.0  (override arch detection)
  - TORCH_BLAS_PREFER_CUTLASS=1  (use CUTLASS instead of rocBLAS)
  - ROCBLAS_LAYOUT=               (force row-major/col-major)
  - HIP_VISIBLE_DEVICES=0    (single GPU, no peer access)
  - RCCL_DEBUG=INFO          (diagnostics)
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False  (disable frag)
  - No env vars (baseline)

Each combination runs the same transcribe test 3 times.

Usage:
    .venv/bin/python packages/research/whisper-rocm-hang/test_03_env_vars.py
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_rocm_hang.setup import device_info, _REPO_ROOT


# The Python code executed inside each subprocess
_INNER_CODE = r"""
import os, sys
sys.path.insert(0, '.')

from whisper_rocm_hang.setup import _enable_conv_patch, load_whisper_model
from pathlib import Path

_enable_conv_patch()

model = load_whisper_model("large-v3-turbo", device="cuda")

vocals = Path("workfolder/jawed/Me_at_the_zoo__jNQXAC9IVRw/media/audio_vocals.wav")
audio = __import__("whisper").audio

if vocals.exists():
    result = model.transcribe(str(vocals), language="en", verbose=False)
    text = (result.get("text") or "").strip()
    print(f"OK ({len(text)} chars): {text[:100]}")
else:
    print("audio NOT FOUND")
"""

ENV_COMBOS = [
    # (label, {env_var: value, ...})
    ("baseline (no extra vars)", {}),
    ("HSA_ENABLE_SDMA=0", {"HSA_ENABLE_SDMA": "0"}),
    ("TORCH_BLAS_PREFER_CUTLASS=1", {"TORCH_BLAS_PREFER_CUTLASS": "1"}),
    ("ROCBLAS_LAYOUT=col", {"ROCBLAS_LAYOUT": "col"}),
    ("HSA_ENABLE_SDMA=0 + CUTLASS=1", {"HSA_ENABLE_SDMA": "0", "TORCH_BLAS_PREFER_CUTLASS": "1"}),
    ("expandable_segments:False", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}),
    ("HIP_VISIBLE_DEVICES=0", {"HIP_VISIBLE_DEVICES": "0"}),
    ("HSA_OVERRIDE_GFX_VERSION=11.0.0 (only)", {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}),
]


def _run_one(combo_label: str, env_add: dict, trial: int) -> tuple[str, float]:
    python_bin = str(_REPO_ROOT / ".venv" / "bin" / "python")
    base_env = os.environ.copy()
    # strip any conflicting vars from base for clean test
    for k in env_add:
        base_env.pop(k, None)
    full_env = {**base_env, **env_add}

    t0 = time.time()
    try:
        proc = subprocess.run(
            [python_bin, "-c", _INNER_CODE],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=_REPO_ROOT,
            env=full_env,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            code = proc.returncode
            if code == -6:
                return ("HANG", elapsed)
            elif code == -11:
                return ("HANG", elapsed)
            else:
                return (f"ERR({code})", elapsed)
        else:
            return ("pass", elapsed)
    except subprocess.TimeoutExpired:
        return ("HANG(timeout)", time.time() - t0)


def main():
    print("=== test_03_env_vars: Bypass via environment ===")
    print(f"Device info: {device_info()}")
    print(f"Combinations: {len(ENV_COMBOS)}")
    print(f"Trials per combo: 3")

    for label, env_add in ENV_COMBOS:
        passes = hangs = errors = 0
        durations = []
        print(f"\n  {label}:")
        for t in range(3):
            status, dur = _run_one(label, env_add, t)
            durations.append(dur)
            if status == "pass":
                passes += 1
                print(f"    trial {t + 1}: pass  ({dur:.1f}s)")
            elif "HANG" in status:
                hangs += 1
                print(f"    trial {t + 1}: HANG  ({dur:.1f}s)")
            else:
                errors += 1
                print(f"    trial {t + 1}: {status}  ({dur:.1f}s)")
        total = passes + hangs + errors
        pass_pct = passes / total * 100 if total else 0
        print(f"    => pass={passes}/{total} ({pass_pct:.0f}%), hang={hangs}/{total}")


if __name__ == "__main__":
    main()
