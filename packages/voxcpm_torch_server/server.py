"""
Standalone Gradio server for local VoxCPM TTS.
Matches the /generate endpoint signature of voxcpm.modelbest.cn.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np

_VOXCPM: "VoxCPM | None" = None
_VOXCPM_LOAD_TIME: float = 0.0
_MODEL_STATUS: str = "unloaded"  # "unloaded" | "loading" | "ready" | "error"
_MODEL_DEVICE: str = ""
_START_TIME: float = time.time()

REPO_ROOT = Path(__file__).resolve().parents[2]

from servers_py.mdns_server import register_service, unregister_service  # noqa: E402


def load_model(model_dir: str, device: str) -> None:
    global _VOXCPM, _VOXCPM_LOAD_TIME, _MODEL_STATUS, _MODEL_DEVICE
    if _VOXCPM is not None:
        return

    _MODEL_STATUS = "loading"
    _MODEL_DEVICE = device
    voxcpm_src = REPO_ROOT / "submodule" / "VoxCPM" / "src"
    if str(voxcpm_src) not in sys.path:
        sys.path.insert(0, str(voxcpm_src))

    if not os.path.isdir(model_dir) or not os.listdir(model_dir):
        sys.stderr.write(f"[VoxCPM] Model not found at {model_dir}, attempting download...\n")
        sys.stderr.flush()
        try:
            from modelscope import snapshot_download
            os.makedirs(model_dir, exist_ok=True)
            snapshot_download("OpenBMB/VoxCPM2", local_dir=model_dir)
            sys.stderr.write("[VoxCPM] Model downloaded via ModelScope\n")
        except Exception as exc:
            sys.stderr.write(f"[VoxCPM] ModelScope download failed ({exc}), trying HuggingFace\n")
        sys.stderr.flush()

    from voxcpm import VoxCPM
    t0 = time.perf_counter()
    try:
        _VOXCPM = VoxCPM.from_pretrained(model_dir, load_denoiser=False, device=device)
        _VOXCPM_LOAD_TIME = time.perf_counter() - t0
        _MODEL_STATUS = "ready"
        sys.stderr.write(f"[VoxCPM] Model loaded ({_VOXCPM_LOAD_TIME:.1f}s, device={device})\n")
    except Exception as e:
        _MODEL_STATUS = "error"
        sys.stderr.write(f"[VoxCPM] Model loading failed: {e}\n")
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
    if _MODEL_STATUS != "ready":
        raise RuntimeError(f"Model not ready (status={_MODEL_STATUS}). Call POST /load-model first.")

    ref_path = None
    if reference_audio and isinstance(reference_audio, str) and os.path.isfile(reference_audio):
        ref_path = reference_audio

    wav = _VOXCPM.generate(
        text=text,
        reference_wav_path=ref_path,
        cfg_value=cfg_value,
        inference_timesteps=dit_steps,
    )

    return 48000, wav.astype(np.float32)


# FastAPI + Gradio app (built at module level for uvicorn reload compat)

# Module-level variables set by main() so endpoints can reference them
_HOST = "127.0.0.1"
_PORT = 19112
_DEVICE = "cpu"
_MODEL_DIR = str(REPO_ROOT / "data" / "models" / "voxcpm2")


def build_app() -> "FastAPI":
    """Create the FastAPI app with custom endpoints and Gradio interface."""
    import gradio as gr
    from fastapi import FastAPI

    fastapi_app = FastAPI(title="VoxCPM Torch Gradio")

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
        load_model(_MODEL_DIR, device)
        return {"status": "ok", "model_loaded": True, "device": device}

    @fastapi_app.post("/unload-model")
    async def unload_model_endpoint():
        global _VOXCPM, _MODEL_STATUS
        _VOXCPM = None
        _MODEL_STATUS = "unloaded"
        import gc
        gc.collect()
        return {"status": "ok", "model_unloaded": True}

    server_port = _PORT
    server_host = _HOST

    # Try to find a free port if the default is occupied
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((server_host, server_port))
        sock.close()
    except OSError:
        server_port = 0
    finally:
        sock.close()

    with gr.Blocks(title="VoxCPM Local") as iface:
        gr.Markdown(f"# VoxCPM Local — device={_DEVICE}")

        with gr.Row():
            status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            load_btn = gr.Button("⬇ Load Model", variant="primary")
            unload_btn = gr.Button("⏫ Unload Model", variant="stop")
            model_status = gr.Textbox(label="Model Status", value=_MODEL_STATUS, interactive=False)

        async def refresh_status():
            return _MODEL_STATUS

        async def handle_load_click():
            try:
                from concurrent.futures import ThreadPoolExecutor
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    await loop.run_in_executor(pool, lambda: load_model(_MODEL_DIR, _DEVICE))
                return "ready"
            except Exception as e:
                return f"error: {e}"

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

    fastapi_app = gr.mount_gradio_app(fastapi_app, iface, path="/")
    return fastapi_app


_uvicorn_app = build_app()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Local VoxCPM Gradio server")
    parser.add_argument("--port", type=int, default=19112, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "models" / "voxcpm2"),
        help="VoxCPM model directory",
    )
    args = parser.parse_args()

    global _HOST, _PORT, _DEVICE, _MODEL_DIR
    _HOST = args.host
    _PORT = args.port
    _DEVICE = args.device
    _MODEL_DIR = args.model_dir

    # Register mDNS so CLI/UI can discover the actual port
    _mdns_zc = register_service("voxcpm_torch_gradio", args.port)

    import uvicorn

    print(f"PORT={args.port}", flush=True)
    print(f"[VoxCPM] Starting Gradio server on {args.host}:{args.port}", flush=True)

    try:
        if args.reload:
            uvicorn.run("server:_uvicorn_app", host=args.host, port=args.port, log_level="info", reload=True)
        else:
            uvicorn.run(_uvicorn_app, host=args.host, port=args.port, log_level="info")
    finally:
        unregister_service(_mdns_zc)


if __name__ == "__main__":
    main()
