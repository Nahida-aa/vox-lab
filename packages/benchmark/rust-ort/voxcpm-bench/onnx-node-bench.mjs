import * as ort from "onnxruntime-node";
import { performance } from "node:perf_hooks";

const MODEL_DIR = process.env.HOME + "/repos/learn_ls/YouDub-webui/data/models/OpenBMB__VoxCPM2";
const WARMUP = 2;
const ITERS = 5;

function randn(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.random() * 2 - 1;
  return a;
}

async function main() {
  console.log("=== onnxruntime-node vs Rust ort benchmark ===\n");

  // --- VAE Encoder: [1, 1, 160000] -> [1, 64, 250] ---
  console.log("[VAE Encoder] Input: [1, 1, 160000]");
  const t0 = performance.now();
  const enc = await ort.InferenceSession.create(MODEL_DIR + "/audio_vae_encoder.onnx", {
    executionProviders: ["cpu"],
  });
  console.log(`  Load: ${((performance.now() - t0) / 1000).toFixed(3)}s`);

  const audioData = randn(160000);
  const audioTensor = new ort.Tensor("float32", audioData, [1, 1, 160000]);

  for (let i = 0; i < WARMUP; i++) {
    await enc.run({ audio_data: audioTensor });
  }
  const t1 = performance.now();
  for (let i = 0; i < ITERS; i++) {
    await enc.run({ audio_data: audioTensor });
  }
  const encMs = (performance.now() - t1) / ITERS;
  console.log(`  Inference: ${encMs.toFixed(1)}ms avg (${ITERS} iters)`);

  // --- VAE Decoder: [1, 64, 2000] -> [1, 1, 3840000] ---
  console.log("\n[VAE Decoder] Input: [1, 64, 2000]");
  const t2 = performance.now();
  const dec = await ort.InferenceSession.create(MODEL_DIR + "/audio_vae_decoder.onnx", {
    executionProviders: ["cpu"],
  });
  console.log(`  Load: ${((performance.now() - t2) / 1000).toFixed(3)}s`);

  const zData = new Float32Array(128000);
  for (let i = 0; i < 128000; i++) zData[i] = Math.random() * 0.1;
  const zTensor = new ort.Tensor("float32", zData, [1, 64, 2000]);

  for (let i = 0; i < WARMUP; i++) {
    await dec.run({ z: zTensor });
  }
  const t3 = performance.now();
  for (let i = 0; i < ITERS; i++) {
    await dec.run({ z: zTensor });
  }
  const decMs = (performance.now() - t3) / ITERS;
  console.log(`  Inference: ${decMs.toFixed(1)}ms avg (${ITERS} iters)`);

  console.log("\n--- Results ---");
  console.log(`onnxruntime-node VAE Encoder: ${encMs.toFixed(1)}ms`);
  console.log(`onnxruntime-node VAE Decoder: ${decMs.toFixed(1)}ms`);
}

main().catch(console.error);
