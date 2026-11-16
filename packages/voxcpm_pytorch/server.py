"""server — 本地 Gradio + FastAPI 服务，复刻云端 /generate 签名。

只做编排：模型加载与推理都委托给 `core`。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

from . import core

_VOXCPM = None
_VOXCPM_LOAD_TIME: float = 0.0
_MODEL_STATUS: str = "unloaded"  # "unloaded" | "loading" | "ready" | "error"
_MODEL_DEVICE: str = ""
_START_TIME: float = time.time()

REPO_ROOT = core.REPO_ROOT

from servers_py.mdns_server import register_service, unregister_service  # noqa: E402


def load_model(model_dir: str, device: str) -> None:
    global _VOXCPM, _VOXCPM_LOAD_TIME, _MODEL_STATUS, _MODEL_DEVICE
    if _VOXCPM is not None:
        return

    _MODEL_STATUS = "loading"
    _MODEL_DEVICE = device
    try:
        _VOXCPM = core.load_model(model_dir, device=device)
        _VOXCPM_LOAD_TIME = float(getattr(_VOXCPM, "_vox_lab_load_s", 0.0))
        _MODEL_STATUS = "ready"
        sys.stderr.write(f"[VoxCPM] Model loaded ({_VOXCPM_LOAD_TIME:.1f}s, device={device})\n")
    except Exception as exc:  # keep the endpoint usable, report via /status
        _MODEL_STATUS = "error"
        sys.stderr.write(f"[VoxCPM] Model loading failed: {exc}\n")
        raise
    sys.stderr.flush()


def generate(
    text: str,
    control_instruction: str = "",
    reference_audio: str | None = None,
    ultimate: bool = False,
    prompt_text: str = "",
    cfg_value: float = 2.0,
    normalize: bool = False,
    ref_denoise: bool = False,
    dit_steps: int = 10,
    _: str = "",
) -> tuple[int, np.ndarray]:
    """VoxCPM generate endpoint — same signature as voxcpm.modelbest.cn."""
    if _MODEL_STATUS != "ready" or _VOXCPM is None:
        raise RuntimeError(f"Model not ready (status={_MODEL_STATUS}). Call POST /load-model first.")

    ref_path = None
    if reference_audio and isinstance(reference_audio, str) and Path(reference_audio).is_file():
        ref_path = reference_audio

    wav = core.generate(
        _VOXCPM,
        text,
        reference_wav=ref_path,
        cfg_value=cfg_value,
        dit_steps=dit_steps,
    )
    return core.DEFAULT_SAMPLE_RATE, wav


# Module-level variables set by main() so endpoints can reference them
_HOST = "127.0.0.1"
_PORT = 19112
_DEVICE = "cpu"
_MODEL_DIR = str(core.DEFAULT_MODEL_DIR)


def build_app():
    """Create the FastAPI app with custom endpoints and Gradio interface."""
    import gradio as gr
    from fastapi import FastAPI

    fastapi_app = FastAPI(title="VoxCPM PyTorch Gradio")

    @fastapi_app.on_event("startup")
    async def on_startup():
        print("[VoxCPM] Ready", flush=True)

    @fastapi_app.get("/status")
    async def status():
        return {
            "status": "running",
            "port": _PORT,
            "uptime_s": int(time.time() - _START_TIME),
            "models": {"voxcpm": {"status": _MODEL_STATUS, "device": _MODEL_DEVICE}},
        }

    @fastapi_app.post("/load-model")
    async def load_model_endpoint(body: dict | None = None):
        device = (body or {}).get("device") or _DEVICE
        sys.stderr.write(f"[VoxCPM] Loading model on {device}...\n")
        sys.stderr.flush()
        await asyncio.get_running_loop().run_in_executor(None, load_model, _MODEL_DIR, device)
        return {"status": "ok", "model_loaded": True, "device": device}

    @fastapi_app.post("/unload-model")
    async def unload_model_endpoint():
        global _VOXCPM, _MODEL_STATUS
        _VOXCPM = None
        _MODEL_STATUS = "unloaded"
        import gc

        gc.collect()
        return {"status": "ok", "model_unloaded": True}

    with gr.Blocks(title="VoxCPM Local") as iface:
        gr.Markdown(f"# VoxCPM Local — device={_DEVICE}")

        with gr.Row():
            status_btn = gr.Button("刷新状态", variant="secondary")
            load_btn = gr.Button("加载模型", variant="primary")
            unload_btn = gr.Button("卸载模型", variant="stop")
            model_status = gr.Textbox(label="Model Status", value=_MODEL_STATUS, interactive=False)

        async def refresh_status():
            return _MODEL_STATUS

        async def handle_load_click():
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, load_model, _MODEL_DIR, _DEVICE)
                return "ready"
            except Exception as exc:
                return f"error: {exc}"

        async def handle_unload_click():
            global _VOXCPM, _MODEL_STATUS
            _VOXCPM = None
            _MODEL_STATUS = "unloaded"
            import gc

            gc.collect()
            return "unloaded"

        status_btn.click(fn=refresh_status, outputs=model_status)
        load_btn.click(fn=handle_load_click, outputs=model_status)
        unload_btn.click(fn=handle_unload_click, outputs=model_status)

        gr.Markdown("---")
        gr.Interface(
            fn=generate,
            inputs=[
                gr.Textbox(label="text", placeholder="Input text"),
                gr.Textbox(label="control_instruction", value=""),
                gr.Audio(type="filepath", label="reference_audio"),
                gr.Checkbox(label="ultimate", value=False),
                gr.Textbox(label="prompt_text", value=""),
                gr.Slider(0, 5, value=2.0, step=0.1, label="cfg_value"),
                gr.Checkbox(label="normalize", value=False),
                gr.Checkbox(label="ref_denoise", value=False),
                gr.Number(value=10, label="dit_steps", precision=0),
                gr.Textbox(value="", visible=False),
            ],
            outputs=gr.Audio(type="numpy", label="Generated Audio"),
        )

    return gr.mount_gradio_app(fastapi_app, iface, path="/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local VoxCPM Gradio server")
    parser.add_argument("--port", type=int, default=19112)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-dir", type=str, default=str(core.DEFAULT_MODEL_DIR))
    parser.add_argument("--load-on-start", action="store_true", help="启动时立即加载模型")
    args = parser.parse_args()

    global _HOST, _PORT, _DEVICE, _MODEL_DIR
    _HOST, _PORT, _DEVICE, _MODEL_DIR = args.host, args.port, args.device, args.model_dir

    if args.load_on_start:
        load_model(_MODEL_DIR, _DEVICE)

    app = build_app()
    _mdns_zc = register_service("voxcpm_torch_gradio", args.port)

    import uvicorn

    print(f"PORT={args.port}", flush=True)
    print(f"[VoxCPM] Starting Gradio server on {args.host}:{args.port}", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        unregister_service(_mdns_zc)


if __name__ == "__main__":
    main()
