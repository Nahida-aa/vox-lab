"""CosyVoice3 ONNX CPU benchmark — phases: LLM, Flow, HiFT for various input lengths."""

import os
import sys
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
ONNX_DIR = REPO_ROOT / 'data' / 'modelscope' / 'CosyVoice3-0.5B' / 'onnx'
MODEL_DIR = str(ONNX_DIR.parent)
SCRIPTS_DIR = ONNX_DIR / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ONNX_DIR))

RESULTS_DIR = Path(__file__).parent / "results"

PROMPT_WAV = str(ONNX_DIR / 'prompts' / 'en_female_nova_greeting.wav')
PROMPT_TEXT = "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions."

TEXTS = {
    "short": "Hello, how are you?",
    "medium": "Today is a beautiful day. Let's go for a walk in the park and enjoy the sunshine together.",
    "long": "Artificial intelligence is transforming the way we live and work. From natural language processing to computer vision and autonomous driving, AI technologies have made remarkable progress in recent years. These advances are creating new opportunities and challenges across every industry.",
}

def run(text_key: str = "medium") -> dict:
    from onnx_inference_pure import PureOnnxCosyVoice3

    text = TEXTS[text_key]

    t0 = time.perf_counter()
    engine = PureOnnxCosyVoice3(MODEL_DIR, use_fp16=True)
    load_time = time.perf_counter() - t0

    SR = 24000

    t0 = time.perf_counter()
    audio = engine.inference(text, PROMPT_WAV, PROMPT_TEXT)
    gen_time = time.perf_counter() - t0

    out_dur = len(audio) / SR
    return {
        "engine": "python",
        "device": "cpu",
        "text_key": text_key,
        "text_len": len(text),
        "load_time_s": round(load_time, 3),
        "generate_time_s": round(gen_time, 3),
        "total_time_s": round(load_time + gen_time, 3),
        "output_samples": len(audio),
        "output_duration_s": round(out_dur, 3),
        "rtf": round(gen_time / out_dur, 3),
    }

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for key in TEXTS:
        print(f"\n=== Benchmarking text=\"{key}\" ===")
        r = run(text_key=key)
        results.append(r)
        print(json.dumps(r, indent=2, ensure_ascii=False))

    summary_path = RESULTS_DIR / "py-cpu.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
