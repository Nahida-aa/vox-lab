"""CLI: compare `demucs-tch` output against the official PyTorch reference.

Usage:
  python -m demucs_pytorch.cli <input.wav> <demucs-tch-output-dir> [--model htdemucs]

Reads the four `target_<i>_<stem>.wav` files produced by demucs-tch, runs the
official PyTorch Demucs on the same input, and reports per-stem error metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from demucs_pytorch import core


def read_stems(out_dir: Path, stem_names: list[str]) -> dict[str, np.ndarray]:
    """Read demucs-tch output WAVs -> {stem: [2, n_samples]}."""
    stems = {}
    for i, name in enumerate(stem_names):
        path = out_dir / f"target_{i}_{name}.wav"
        if not path.exists():
            raise FileNotFoundError(path)
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)  # [T, 2]
        stems[name] = audio.T  # [2, T]
    return stems


def metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    """Per-stem error metrics between two `[2, T]` arrays."""
    n = min(ref.shape[1], got.shape[1])
    a, b = ref[:, :n], got[:, :n]
    diff = a - b
    mse = float(np.mean(diff**2))
    ref_energy = float(np.mean(a**2))
    sdr = 10.0 * np.log10(ref_energy / mse) if mse > 0 else float("inf")
    corr = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "ref_rms": float(np.sqrt(ref_energy)),
        "got_rms": float(np.sqrt(float(np.mean(b**2)))),
        "sdr_db": float(sdr),
        "corr": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Input WAV fed to both implementations")
    parser.add_argument("out_dir", help="demucs-tch output directory")
    parser.add_argument("--model", default="htdemucs", help="Official model name")
    args = parser.parse_args()

    names = core.sources(args.model)
    print(f"Running official PyTorch Demucs ({args.model}) on {args.wav} ...")
    ref = core.separate(args.wav, args.model)  # [S, 2, T]
    print(f"reference shape: {ref.shape}")

    got = read_stems(Path(args.out_dir), names)

    print(f"\n{'stem':8} {'max|d|':>10} {'mean|d|':>10} {'rmse':>10} "
          f"{'ref_rms':>10} {'got_rms':>10} {'SDR(dB)':>9} {'corr':>8}")
    worst_corr = 1.0
    for i, name in enumerate(names):
        m = metrics(ref[i], got[name])
        worst_corr = min(worst_corr, m["corr"])
        print(f"{name:8} {m['max_abs']:10.5f} {m['mean_abs']:10.5f} {m['rmse']:10.5f} "
              f"{m['ref_rms']:10.5f} {m['got_rms']:10.5f} {m['sdr_db']:9.2f} {m['corr']:8.4f}")

    print(f"\nworst correlation: {worst_corr:.4f}")
    if worst_corr < 0.9:
        print("WARNING: low correlation - demucs-tch output likely diverges from PyTorch.")
    else:
        print("OK: outputs are strongly correlated.")


if __name__ == "__main__":
    main()
