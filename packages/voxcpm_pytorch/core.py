"""core — VoxCPM 的加载 / 生成 / 计时 / 精度实验。

这里不依赖 FastAPI 或 Gradio，CLI 与 server 都只调这一层。

背景（docs/decisions/0011）：VoxCPM2 的权重是 4.4 GB bf16，生成是「自回归 LM +
10 步 CFM + AudioVAE 解码」。要判断自研 GPU kernel 是否值得做，先要量出：
  1. 三段各占多少时间（`profile`）
  2. 换成 f16 / f32 的精度代价与耗时差（`dtype`）
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "models" / "voxcpm2"
DEFAULT_SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _ensure_voxcpm_importable() -> None:
    """Put the official submodule on sys.path (only place that touches it)."""
    src = REPO_ROOT / "submodule" / "VoxCPM" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _ensure_torchaudio()


def _ensure_torchaudio() -> None:
    """Install a stub when torchaudio is unavailable.

    torchaudio has no cp314 wheel (checked on the rocm7.2 index and PyPI), while
    `voxcpm` imports it unconditionally. The experiment paths here only do
    text -> audio, so audio file IO is never exercised; the stub keeps imports
    working and fails loudly if something does reach for it.
    """
    try:
        import torchaudio  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import types

    import importlib.machinery

    mod = types.ModuleType("torchaudio")
    mod.__version__ = "0.0.0+vox_lab_stub"
    # transformers probes availability via find_spec, which needs __spec__
    mod.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    mod.__path__ = []  # package marker so torchaudio.functional resolves

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "torchaudio is not installed (no cp314 wheel); this code path needs audio IO"
        )

    functional = types.ModuleType("torchaudio.functional")
    functional.resample = _unavailable
    mod.load = _unavailable
    mod.save = _unavailable
    mod.functional = functional
    functional.__spec__ = importlib.machinery.ModuleSpec("torchaudio.functional", loader=None)
    sys.modules.setdefault("torchaudio", mod)
    sys.modules.setdefault("torchaudio.functional", functional)
    sys.stderr.write("[voxcpm_pytorch] torchaudio missing; using a stub (audio IO disabled)\n")


# ---------------------------------------------------------------------------
# Load / generate
# ---------------------------------------------------------------------------


def _model_dir_with_dtype(model_dir: Path, dtype: str) -> Path:
    """Return a model dir whose config.json declares `dtype`.

    `dtype` lives in config.json and drives the KV cache dtype at construction
    time, so casting after `from_local` has no effect. Rather than monkey-patching
    internals, mirror the dir with symlinks and rewrite just the config.
    """
    import json
    import os

    cache = Path("/tmp") / "vox-lab-voxcpm-dtype" / f"{model_dir.name}-{dtype}"
    cache.mkdir(parents=True, exist_ok=True)
    config_src = model_dir / "config.json"
    config = json.loads(config_src.read_text(encoding="utf-8"))
    config["dtype"] = dtype
    (cache / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for entry in model_dir.iterdir():
        if entry.name == "config.json":
            continue
        dst = cache / entry.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(entry.resolve(), dst)
    return cache


def load_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    device: str = "cpu",
    optimize: bool = False,
    dtype: str | None = None,
):
    """Load VoxCPM.

    `optimize=False` because `torch.compile` hides the real per-op cost and
    ruins profiling; turn it on only for throughput measurements.
    `dtype` casts the underlying `tts_model` after load (defaults to whatever
    the checkpoint uses, currently bf16).
    """
    _ensure_voxcpm_importable()
    from voxcpm import VoxCPM

    src_dir = Path(model_dir)
    if dtype is not None:
        normalized = {"fp32": "float32", "f32": "float32", "fp16": "float16",
                      "f16": "float16", "bf16": "bfloat16"}.get(dtype.lower(), dtype)
        src_dir = _model_dir_with_dtype(src_dir, normalized)

    t0 = time.perf_counter()
    model = VoxCPM.from_pretrained(
        str(src_dir),
        load_denoiser=False,
        device=device,
        optimize=optimize,
    )
    load_s = time.perf_counter() - t0

    model._vox_lab_load_s = load_s  # type: ignore[attr-defined]
    return model


def generate(
    model,
    text: str,
    *,
    reference_wav: str | None = None,
    cfg_value: float = 2.0,
    dit_steps: int = 10,
    seed: int | None = 42,
    max_len: int | None = None,
) -> np.ndarray:
    """Deterministic generation (fixed seed by default) returning float32 audio.

    `max_len` 限制自回归 token 数。官方默认 4096，一旦 EOS 没被采样出来就是
    小时级；实验里必须给上界。
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    kwargs = {}
    if max_len is not None:
        kwargs["max_len"] = max_len
    wav = model.generate(
        text=text,
        reference_wav_path=reference_wav,
        cfg_value=cfg_value,
        inference_timesteps=dit_steps,
        **kwargs,
    )
    return np.asarray(wav, dtype=np.float32)


def timed_generate(model, text: str, rounds: int = 1, **kwargs) -> tuple[np.ndarray, list[float]]:
    times: list[float] = []
    wav = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        wav = generate(model, text, **kwargs)
        times.append(time.perf_counter() - t0)
    assert wav is not None
    return wav, times


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


@dataclass
class StageTimes:
    """Aggregated self-CPU time per pipeline stage, in milliseconds."""

    stages: dict[str, float]
    total_ms: float
    wall_s: float = 0.0
    audio_s: float = 0.0

    @property
    def rtf(self) -> float:
        return self.wall_s / self.audio_s if self.audio_s else 0.0

    def rows(self) -> list[tuple[str, float, float]]:
        rows = []
        for name, ms in sorted(self.stages.items(), key=lambda kv: -kv[1]):
            share = 100.0 * ms / self.total_ms if self.total_ms else 0.0
            rows.append((name, ms, share))
        return rows


def _classify(name: str) -> str:
    n = name.lower()
    if "audiovae" in n or "vae" in n or "decoder" in n:
        return "vae"
    if "dit" in n or "cfm" in n:
        return "cfm"
    if "residual" in n:
        return "residual_lm"
    if "minicpm" in n or ".lm" in n or n.startswith("lm"):
        return "lm"
    return "other"


def profiled_generate(model, text: str, **kwargs) -> tuple[np.ndarray, StageTimes]:
    """Time the top-level submodules of `tts_model` with forward hooks.

    Kernel-level profiling (torch.profiler) can't tell which stage a kernel
    belongs to: every aten::linear looks the same whether it came from the LM,
    the DiT or the VAE. Hooking the direct children of `tts_model` gives a
    disjoint partition, which is what the "where is the time going" question
    actually needs.
    """
    import torch

    root = model.tts_model
    stacks: dict[str, list[float]] = {}
    stages: dict[str, float] = {}
    handles = []

    def make_hooks(name: str):
        def pre(_module, _inputs):
            stacks.setdefault(name, []).append(time.perf_counter())

        def post(_module, _inputs, _output):
            started = stacks[name].pop()
            stages[name] = stages.get(name, 0.0) + (time.perf_counter() - started)

        return pre, post

    # Hook leaves only, then attribute each leaf to its top-level group.
    # Containers can be called through a submodule (e.g. audio_vae.decoder(x)),
    # which never triggers a hook on the container itself, so hooking the
    # top-level children alone leaves large blocks unattributed.
    for child_name, module in root.named_modules():
        if not child_name or next(module.children(), None) is not None:
            continue
        pre, post = make_hooks(child_name.split(".")[0])
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        wav = generate(model, text, **kwargs)
    finally:
        for h in handles:
            h.remove()
    wall_s = time.perf_counter() - t0

    # tensors are still lazy on GPU; a second sync makes the wall time honest
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    stage_ms = {name: ms * 1000.0 for name, ms in stages.items()}
    return wav, StageTimes(
        stages=stage_ms,
        total_ms=sum(stage_ms.values()),
        wall_s=wall_s,
        audio_s=len(wav) / DEFAULT_SAMPLE_RATE,
    )


def stage_breakdown(model, text: str, **kwargs) -> StageTimes:
    _, times = profiled_generate(model, text, **kwargs)
    return times


# ---------------------------------------------------------------------------
# dtype experiment
# ---------------------------------------------------------------------------


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    n = min(len(reference), len(candidate))
    ref, got = reference[:n], candidate[:n]
    noise = got - ref
    denom = float(np.mean(noise**2))
    if denom == 0:
        return float("inf")
    return 10.0 * np.log10(float(np.mean(ref**2)) / denom)


@dataclass
class DtypeResult:
    dtype: str
    load_s: float
    gen_s: float
    snr_db: float
    samples: int


def dtype_experiment(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    text: str = "这是一段用于精度对比的测试语音。",
    device: str = "cpu",
    dtypes: tuple[str, ...] = ("bf16", "fp32"),
    dit_steps: int = 10,
    max_len: int = 120,
) -> list[DtypeResult]:
    """Generate the same sentence once per dtype and compare against the first.

    Models are loaded one at a time and freed immediately: VoxCPM2 is ~4.4 GB
    in bf16 and ~8.7 GB in f32, so holding two copies is not an option on most
    machines.
    """
    results: list[DtypeResult] = []
    reference: np.ndarray | None = None

    for dtype in dtypes:
        model = load_model(model_dir, device=device, optimize=False, dtype=dtype)
        wav, times = timed_generate(model, text, rounds=1, dit_steps=dit_steps, max_len=max_len)
        result = DtypeResult(
            dtype=dtype,
            load_s=float(getattr(model, "_vox_lab_load_s", 0.0)),
            gen_s=times[0],
            snr_db=0.0,
            samples=len(wav),
        )
        if reference is None:
            reference = wav
            result.snr_db = float("inf")
        else:
            result.snr_db = _snr_db(reference, wav)
        results.append(result)

        del model
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    return results


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    import soundfile as sf

    sf.write(str(path), audio, sample_rate)
