"""CosyVoice2 PyTorch CPU benchmark — times model load + inference for various input lengths."""

import os
import sys
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
COSYVOICE_DIR = REPO_ROOT / 'submodule' / 'CosyVoice'
MATCHA_DIR = COSYVOICE_DIR / 'third_party' / 'Matcha-TTS'
MODEL_DIR = REPO_ROOT / 'pretrained_models' / 'CosyVoice2-0.5B'
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(MATCHA_DIR))

import torch
torch.set_default_dtype(torch.float32)

REF_WAV = Path(__file__).parent / "ref.wav"
RESULTS_DIR = Path(__file__).parent / "results"

TEXTS = {
    "short": "你好。",
    "medium": "今天天气真不错，我们一起去公园散步吧。",
    "long": "请播放一段关于人工智能发展的新闻。近年来，人工智能技术在各个领域都取得了显著的进展，从自然语言处理到计算机视觉，再到自动驾驶，AI正在改变我们的生活方式。",
}

PROMPT_TEXT = "希望你以后能够做的比我还好呦。"


def run(text_key: str = "medium") -> dict:
    from cosyvoice.cli.cosyvoice import CosyVoice2

    text = TEXTS[text_key]

    t0 = time.perf_counter()
    model = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        for i, j in enumerate(model.inference_zero_shot(text, PROMPT_TEXT, str(REF_WAV), stream=False)):
            wav = j['tts_speech']
    gen_time = time.perf_counter() - t0

    return {
        "engine": "python",
        "device": "cpu",
        "text_key": text_key,
        "text_len": len(text),
        "load_time_s": round(load_time, 3),
        "generate_time_s": round(gen_time, 3),
        "total_time_s": round(load_time + gen_time, 3),
        "output_samples": wav.shape[-1],
        "output_duration_s": round(wav.shape[-1] / model.sample_rate, 3),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for key in TEXTS:
        print(f"\nBenchmarking text=\"{key}\"...")
        r = run(text_key=key)
        results.append(r)
        print(json.dumps(r, indent=2, ensure_ascii=False))

    summary_path = RESULTS_DIR / "py-cpu.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
