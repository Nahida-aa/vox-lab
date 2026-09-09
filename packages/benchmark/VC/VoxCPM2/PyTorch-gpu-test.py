"""Test Python VoxCPM on GPU — does it actually hang?"""

import os, sys, time, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend'))

from app.config import MODEL_CACHE_DIR, DATA_DIR
from app.adapters.voxcpm import _model_path
from pathlib import Path

REF_WAV = Path(__file__).parent / "ref.wav"

def test_gpu(timesteps: int = 3, timeout: int = 120):
    model_dir = _model_path()
    print(f"[test_gpu] Model dir: {model_dir}")
    print(f"[test_gpu] GPU available: {__import__('torch').cuda.is_available()}")

    # Import & load on GPU
    from voxcpm import VoxCPM
    t0 = time.time()
    model = VoxCPM.from_pretrained(
        str(model_dir),
        load_denoiser=False,
        device="cuda",
    )
    t1 = time.time()
    print(f"[test_gpu] Load time: {t1-t0:.1f}s")

    # Verify it's actually on GPU
    model_device = next(model.tts_model.parameters()).device
    print(f"[test_gpu] Model device: {model_device}")

    # Minimal generation
    text = "测试GPU推理。"
    if not REF_WAV.exists():
        print(f"[test_gpu] ERROR: ref.wav not found at {REF_WAV}")

    t2 = time.time()
    try:
        wav = model.generate(
            text=text,
            reference_wav_path=str(REF_WAV),
            cfg_value=2.0,
            inference_timesteps=timesteps,
        )
        t3 = time.time()
        print(f"[test_gpu] Generation OK: {len(wav)} samples, {t3-t2:.1f}s ({timesteps} timesteps)")
        return True
    except Exception as e:
        t3 = time.time()
        print(f"[test_gpu] Generation FAILED after {t3-t2:.1f}s: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3, help="Inference timesteps (default 3)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    args = parser.parse_args()
    ok = test_gpu(timesteps=args.timesteps, timeout=args.timeout)
    sys.exit(0 if ok else 1)
