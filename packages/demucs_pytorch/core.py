"""Core: load the official PyTorch Demucs model and produce reference stems.

This package exists to validate `packages/demucs-tch` (the Burn-free tch-rs
port) against the upstream PyTorch implementation. It only serves Demucs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_DEMUCS = REPO_ROOT / "submodule" / "demucs"
if str(_OFFICIAL_DEMUCS) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_DEMUCS))

# Imported lazily so that importing `demucs_pytorch.core` does not require
# torch / the official demucs package until a model is actually loaded.
_model_cache: dict[str, object] = {}


def load_model(name: str = "htdemucs"):
    """Load (and cache) an official pretrained Demucs model by name."""
    if name not in _model_cache:
        import torch
        from demucs.pretrained import get_model

        model = get_model(name)
        model.eval()
        _model_cache[name] = (model, torch)
    return _model_cache[name]


def sources(name: str = "htdemucs") -> list[str]:
    """Ordered stem names for a model."""
    model, _ = load_model(name)
    return list(model.sources)


def separate(wav_path: str | Path, model_name: str = "htdemucs") -> np.ndarray:
    """Run official PyTorch Demucs on a WAV file.

    Returns a float32 numpy array of shape `[n_sources, 2, n_samples]` at the
    input sample rate (resampling back handled by `demucs.apply.apply_model`).
    """
    import soundfile as sf
    from demucs.apply import apply_model

    model, torch = load_model(model_name)
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # [T, C]
    x = torch.from_numpy(audio.T).unsqueeze(0)  # [1, C, T]
    if x.shape[1] == 1:
        x = x.repeat(1, 2, 1)
    elif x.shape[1] > 2:
        x = x[:, :2, :]

    with torch.no_grad():
        out = apply_model(model, x, device="cpu", progress=False, num_workers=0)[0]
    return out.cpu().numpy().astype(np.float32)
