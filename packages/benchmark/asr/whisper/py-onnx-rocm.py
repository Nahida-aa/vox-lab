"""Whisper ASR benchmark: Python ONNX (ROCm EP fallback).

ROCm EP fails at runtime: libonnxruntime_providers_rocm.so 1.22.2
links to ROCm 5.x symbols (hipblasGemmStridedBatchedEx_v2) not in ROCm 7.2.
Falls back to CPU EP.
"""
import json, math, os, struct, subprocess, sys, time
from pathlib import Path

import numpy as np
import onnxruntime

MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" /  "models" / "sherpa-whisper-turbo"
WAV = "/tmp/chirp.wav"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

VOCAB_SIZE = 51866
DECODER_START_TOKEN = 50258
EOS_TOKEN = 50257
SR = 16000
N_FFT = 400
HOP = 160
N_MEL = 128
MAX_SEQ = 448
D_MODEL = 1280
N_LAYERS = 4


def hann_window(n):
    return np.array([0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)], dtype=np.float64)


_hann = hann_window(N_FFT)


def mel_filterbank():
    n_fft = N_FFT
    f_min, f_max = 0, SR / 2
    mel_min = 2595 * math.log10(1 + f_min / 700)
    mel_max = 2595 * math.log10(1 + f_max / 700)
    pts = np.linspace(mel_min, mel_max, N_MEL + 2)
    hz = 700 * (10 ** (pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz / SR).astype(int)
    banks = np.zeros((N_MEL, n_fft // 2 + 1), dtype=np.float64)
    for i in range(N_MEL):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            banks[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            banks[i, c:r + 1] = (r - np.arange(c, r + 1)) / (r - c)
    return banks


_banks = mel_filterbank()


def compute_log_mel(pcm):
    n_frames = max((len(pcm) - N_FFT + HOP) // HOP, 0)
    mel = np.zeros((N_MEL, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * HOP
        frame = pcm[start:start + N_FFT] if start + N_FFT <= len(pcm) else np.pad(pcm[start:], (0, N_FFT - len(pcm[start:])))
        frame = frame * _hann
        mag2 = np.abs(np.fft.rfft(frame, n=N_FFT)) ** 2
        for m in range(N_MEL):
            mel[m, i] = math.log10(max(np.dot(mag2, _banks[m]), 1e-10))
    for m in range(N_MEL):
        mean, std = mel[m].mean(), mel[m].std()
        if std > 0:
            mel[m] = (mel[m] - mean) / std
    out = np.zeros((N_MEL, 3000), dtype=np.float32)
    out[:, :n_frames] = mel[:, :n_frames]
    return out[np.newaxis, :, :]


def load_audio(path):
    r = subprocess.run(
        ["ffmpeg", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(r.stdout, dtype=np.float32)


def load_tokens():
    text = (MODEL_DIR / "turbo-tokens.txt").read_text().strip().split("\n")
    return text


def tokens_to_text(tokens, vocab):
    buf = bytearray()
    for tid in tokens:
        if tid < 0 or tid >= len(vocab):
            continue
        t = vocab[tid]
        if not t or "<|" in t or "|>" in t:
            continue
        buf.extend(base64_decode(t))
    return buf.decode("utf-8", errors="replace").strip()


BASE64_TABLE = {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")}
BASE64_TABLE["="] = 0


def base64_decode(s):
    bits = 0
    acc = 0
    out = bytearray()
    for c in s:
        if c == "=":
            break
        v = BASE64_TABLE.get(c, 0)
        acc = (acc << 6) | v
        bits += 6
        if bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
    return bytes(out)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    providers = onnxruntime.get_available_providers()
    print(f"=== Whisper ONNX ROCm Benchmark ===")
    print(f"ORT: {onnxruntime.__version__}, providers: {providers}")

    # Test if ROCm EP actually works (it's listed but may fail at session creation)
    ep_available = False
    if "ROCMExecutionProvider" in providers:
        try:
            test_sess = onnxruntime.InferenceSession(
                str(MODEL_DIR / "turbo-encoder.int8.onnx"),
                providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
            )
            test_sess = None
            ep_available = True
        except Exception as e:
            print(f"  ROCm EP unavailable at runtime: {e.message[:80] if hasattr(e, 'message') else str(e)[:80]}")

    ep = ["ROCMExecutionProvider", "CPUExecutionProvider"] if ep_available else ["CPUExecutionProvider"]
    print(f"EP: {ep}")

    enc_path = str(MODEL_DIR / "turbo-encoder.int8.onnx")
    dec_path = str(MODEL_DIR / "turbo-decoder.int8.onnx")

    t0 = time.perf_counter()
    enc = onnxruntime.InferenceSession(enc_path, providers=ep)
    dec = onnxruntime.InferenceSession(dec_path, providers=ep)
    load_time = time.perf_counter() - t0
    print(f"  load OK  ({load_time:.2f}s)")

    pcm = load_audio(WAV)
    audio_s = len(pcm) / SR
    mel = compute_log_mel(pcm)

    te0 = time.perf_counter()
    enc_out = enc.run(None, {"mel": mel.astype(np.float32)})
    n_layer_cross_k, n_layer_cross_v = enc_out[0], enc_out[1]
    encode_time = (time.perf_counter() - te0) * 1000

    self_k = np.zeros((N_LAYERS, 1, MAX_SEQ, D_MODEL), dtype=np.float32)
    self_v = np.zeros((N_LAYERS, 1, MAX_SEQ, D_MODEL), dtype=np.float32)
    tokens = []

    td0 = time.perf_counter()
    for step in range(MAX_SEQ):
        token_id = DECODER_START_TOKEN if step == 0 else tokens[-1]
        feeds = {
            "tokens": np.array([[token_id]], dtype=np.int64),
            "in_n_layer_self_k_cache": self_k,
            "in_n_layer_self_v_cache": self_v,
            "n_layer_cross_k": n_layer_cross_k,
            "n_layer_cross_v": n_layer_cross_v,
            "offset": np.array([step], dtype=np.int64),
        }
        out = dec.run(None, feeds)
        logits = out[0][0, -1, :VOCAB_SIZE]
        nt = int(np.argmax(logits))
        if nt == EOS_TOKEN:
            break
        tokens.append(nt)
        self_k, self_v = out[1], out[2]
    decode_time = (time.perf_counter() - td0) * 1000

    total_time = time.perf_counter() - t0
    rtf = total_time / audio_s

    vocab = load_tokens()
    text = tokens_to_text(tokens, vocab)

    result = {
        "engine": "onnx",
        "device": "rocm" if ep_available else "cpu",
        "ort_version": onnxruntime.__version__,
        "ep": ep[0],
        "rocm_ep_available": ep_available,
        "load_time_s": round(load_time, 3),
        "encode_time_ms": round(encode_time, 1),
        "decode_time_ms": round(decode_time, 1),
        "total_s": round(total_time, 3),
        "audio_s": round(audio_s, 3),
        "rtf": round(rtf, 3),
        "tokens": len(tokens),
        "text": text,
    }
    print(f"\n  RTF={rtf:.2f}, {len(tokens)} tokens, audio={audio_s:.2f}s")
    print(f"  text: {text[:80]}")

    path = RESULTS_DIR / "py-onnx-rocm.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
