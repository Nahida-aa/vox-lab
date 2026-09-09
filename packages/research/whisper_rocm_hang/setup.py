"""
Shared bootstrap for Whisper ROCm Hang research tests.

Usage:
    from whisper_rocm_hang.setup import setup, device_info, load_whisper_model

    ctx = setup(use_patch=True, device="try_cuda")
    for i in range(ctx.trials):
        model = load_whisper_model("tiny", ctx.device)
        # ... run test
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path

# Allow importing from backend/ (for conv_patch)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # packages/research/ -> repo root
sys.path.insert(0, str(_REPO_ROOT))


def load_dotenv_safe() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _enable_conv_patch() -> None:
    from backend.app.adapters.conv_patch import apply_patch

    apply_patch()


def device_info() -> dict:
    """Return dict of torch/cuda/hip info without importing torch in parent process."""
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_memory_mb": round(props.total_memory / 1024 / 1024),
                "gpu_cc": f"{props.major}.{props.minor}",
            }
        )
    return info


def load_whisper_model(name: str = "tiny", device: str = "cpu") -> object:
    import whisper
    import torch  # noqa: F401 - ensure torch device is set

    return whisper.load_model(name, device=device)


class HangResult:
    """Result of a single trial."""

    PASS = "pass"
    HANG = "hang"
    ERROR = "error"

    def __init__(self, status: str, duration_s: float, detail: str = ""):
        self.status = status
        self.duration_s = duration_s
        self.detail = detail

    def __repr__(self) -> str:
        return f"[{self.status.upper():5s}] {self.duration_s:.1f}s  {self.detail}"


def run_trials(
    fn,
    trials: int = 5,
    timeout: int = 120,
    label: str = "",
) -> list[HangResult]:
    """Run fn in a subprocess `trials` times. fn is a string of Python code.

    Each trial spawns a fresh python subprocess to isolate GPU state.
    Returns list of HangResult.
    """
    python_bin = os.environ.get("PYTHON_BIN", str(_REPO_ROOT / ".venv" / "bin" / "python"))
    research_dir = str(_REPO_ROOT / "packages" / "research")
    results: list[HangResult] = []

    for i in range(trials):
        t0 = time.time()
        env = os.environ.copy()
        existing_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{research_dir}:{_REPO_ROOT}" + (f":{existing_pypath}" if existing_pypath else "")
        try:
            proc = subprocess.run(
                [python_bin, "-c", fn],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=_REPO_ROOT,
                env=env,
            )
            elapsed = time.time() - t0
            if proc.returncode == -6 or proc.returncode == -11:
                # SIGABRT (-6) or SIGSEGV (-11) from GPU Hang
                results.append(HangResult(HangResult.HANG, elapsed, f"exit={proc.returncode}"))
            elif proc.returncode != 0:
                stderr = proc.stderr.strip()[-300:] if proc.stderr else ""
                results.append(HangResult(HangResult.ERROR, elapsed, stderr))
            else:
                results.append(HangResult(HangResult.PASS, elapsed, proc.stdout.strip()[-200:]))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            results.append(HangResult(HangResult.HANG, elapsed, "timeout"))
        except Exception as e:
            elapsed = time.time() - t0
            results.append(HangResult(HangResult.ERROR, elapsed, str(e)[-200:]))

        sys.stdout.write(f"  trial {i + 1}/{trials}: {results[-1]}\n")
        sys.stdout.flush()

    return results


def summary_table(results: list[HangResult], label: str = "") -> str:
    passes = sum(1 for r in results if r.status == HangResult.PASS)
    hangs = sum(1 for r in results if r.status == HangResult.HANG)
    errors = sum(1 for r in results if r.status == HangResult.ERROR)
    total = len(results)
    durations = [r.duration_s for r in results if r.status == HangResult.PASS]
    avg_time = sum(durations) / len(durations) if durations else 0.0
    lines = [
        f"── {label} ──" if label else "",
        f"  pass: {passes}/{total}  ({passes / total * 100:.0f}%)",
        f"  hang: {hangs}/{total}  ({hangs / total * 100:.0f}%)",
        f"  error: {errors}/{total}",
        f"  avg pass time: {avg_time:.1f}s" if durations else "",
    ]
    return "\n".join(lines)
