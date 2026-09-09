"""PyTorch VoxCPM benchmark — timesteps=10 (default), reports load+gen+RTF."""
import argparse, json, os, sys, time, wave, numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, REPO + "/submodule/VoxCPM/src")

TEXTS = {
    "short": "今天天气真不错。",
    "medium": "人工智能正在改变我们的生活和工作方式。从自然语言处理到计算机视觉，AI技术取得了显著进展。",
    "long": "近年来，人工智能技术发展迅速，在自然语言处理、计算机视觉、语音识别等领域都取得了突破性进展。深度学习模型的不断完善，使得AI系统在理解和生成人类语言方面表现出色。同时，大语言模型的出现更是推动了整个行业的变革，为各行各业带来了新的机遇和挑战。未来，随着算力的提升和算法的优化，人工智能将在更多领域发挥重要作用。",
}

def benchmark(model_dir: str, ref_wav: str, device: str, timesteps: int, cfg: float, out_dir: str):
    from voxcpm import VoxCPM
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    model = VoxCPM.from_pretrained(model_dir, load_denoiser=False, device=device)
    load_time = time.perf_counter() - t0
    print(f"Load: {load_time:.1f}s", flush=True)

    results = []
    for key, text in TEXTS.items():
        t1 = time.perf_counter()
        wav = model.generate(
            text=text,
            reference_wav_path=ref_wav,
            cfg_value=cfg,
            inference_timesteps=timesteps,
        )
        gen_time = time.perf_counter() - t1
        out_dur = len(wav) / 48000
        rtf = gen_time / out_dur if out_dur > 0 else float("inf")

        wav_i16 = np.clip(wav * 32767, -32768, 32767).astype(np.int16)
        wav_path = f"{out_dir}/pytorch-{key}.wav"
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000)
            wf.writeframes(wav_i16.tobytes())

        r = {
            "engine": "pytorch", "dtype": "f32", "device": device,
            "text_key": key, "text_len": len(text),
            "timesteps": timesteps, "cfg": cfg,
            "load_time_s": round(load_time, 3),
            "generate_time_s": round(gen_time, 3),
            "total_time_s": round(load_time + gen_time, 3),
            "output_samples": len(wav),
            "output_duration_s": round(out_dur, 3),
            "rtf": round(rtf, 3),
        }
        print(json.dumps(r), flush=True)
        results.append(r)

    with open(f"{out_dir}/pytorch.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults -> {out_dir}/pytorch.json", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=REPO + "/data/models/OpenBMB__VoxCPM2")
    p.add_argument("--ref", dest="ref_wav", default=REPO + "/packages/benchmark/VC/VoxCPM2/ref.wav")
    p.add_argument("--device", default="cpu")
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--out-dir", default=REPO + "/packages/benchmark/VC/VoxCPM2/results")
    args = p.parse_args()
    benchmark(**vars(args))
