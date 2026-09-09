import onnxruntime as ort
import numpy as np
import time
import os

MODEL_DIR = "data/models/OpenBMB__VoxCPM2"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

def load_session(model_name, provider="MIGraphXExecutionProvider"):
    path = os.path.join(MODEL_DIR, model_name)
    return ort.InferenceSession(path, providers=[provider])

def main():
    print("Loading models...")
    t0 = time.time()
    enc = load_session("audio_vae_encoder.onnx", provider="CPUExecutionProvider")
    dec = load_session("audio_vae_decoder.onnx")
    prefill = load_session("voxcpm2_prefill.onnx")
    decode = load_session("voxcpm2_decode_step.onnx")
    print(f"All models loaded in {time.time() - t0:.1f}s")

    print(f"ort: {ort.__version__}, providers: {ort.get_available_providers()}")

    # VAE Encoder (CPU)
    dummy_audio = np.random.randn(1, 1, 48000).astype(np.float32)
    t0 = time.time()
    z = enc.run(None, {"audio_data": dummy_audio})[0]
    t1 = time.time()
    print(f"VAE Encoder (CPU): {t1-t0:.3f}s, {z.shape}")

    # Prefill (GPU)
    text = np.array([[1, 1]], dtype=np.int64)
    mask = np.ones((1, 2), dtype=np.int32)
    feat = np.zeros((1, 2, 4, 64), dtype=np.float32)
    feat_mask = np.ones((1, 2), dtype=np.int32)
    t0 = time.time()
    pf_out = prefill.run(None, {"text": text, "text_mask": mask, "feat": feat, "feat_mask": feat_mask})
    t1 = time.time()
    print(f"Prefill: {t1-t0:.1f}s")
    dit_hidden, bnk, bnv, rnk, rnv, pfc = pf_out

    # Decode steps (GPU) - simulate 10 steps
    noise = np.random.randn(1, 4, 64).astype(np.float32)
    cfg = np.array(2.0, dtype=np.float32)
    t0 = time.time()
    for i in range(10):
        ds_out = decode.run(None, {
            "dit_hidden": dit_hidden,
            "base_next_keys": bnk, "base_next_values": bnv,
            "residual_next_keys": rnk, "residual_next_values": rnv,
            "prefix_feat_cond": pfc, "noise": noise, "cfg_value": cfg
        })
        pred_feat, dit_hidden, bnk, bnv, rnk, rnv = ds_out[0], ds_out[1], ds_out[2], ds_out[3], ds_out[4], ds_out[5]
    t1 = time.time()
    print(f"Decode 10 steps: {t1-t0:.3f}s (avg {(t1-t0)/10*1000:.1f}ms/step)")

    # VAE Decoder (GPU)
    t0 = time.time()
    audio = dec.run(None, {"z": z})[0]
    t1 = time.time()
    print(f"VAE Decoder (GPU): {t1-t0:.3f}s, {audio.shape}")
    print("Full pipeline OK")

if __name__ == "__main__":
    main()
