"""Optional HTTP surface for the PyTorch Demucs reference.

Kept minimal by design: the primary use of this package is the CLI comparison
tool (`python -m demucs_pytorch.cli`). A thin FastAPI app is provided so the
reference can also be queried over HTTP if needed.

Run with:
  python -m demucs_pytorch.server --http-port 19110
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from demucs_pytorch import core

app = FastAPI(title="Demucs PyTorch Reference")


@app.get("/health")
async def health():
    return {"status": "ok", "purpose": "pytorch-demucs-reference"}


@app.post("/separate")
async def separate(body: dict):
    """Body: {"wav": "/path/to/input.wav", "model": "htdemucs"}.
    Returns the reference stems as a float32 WAV (4 stems -> 8 channels).
    """
    wav = body.get("wav")
    model_name = body.get("model", "htdemucs")
    if not wav or not Path(wav).exists():
        return JSONResponse({"error": "wav not found"}, status_code=400)
    stems = core.separate(wav, model_name)  # [S, 2, T]
    # Flatten to [T, S*2] interleaved per stem for portable WAV output.
    out = np.transpose(stems, (2, 0, 1)).reshape(stems.shape[2], -1)
    buf = io.BytesIO()
    sf.write(buf, out, 44100, format="WAV", subtype="FLOAT")
    return Response(content=buf.getvalue(), media_type="audio/wav")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-port", type=int, default=19110)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.http_port, log_level="info")


if __name__ == "__main__":
    main()
