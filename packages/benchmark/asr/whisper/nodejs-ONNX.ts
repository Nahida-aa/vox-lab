import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { spawnSync } from "node:child_process";

// ---- Dynamic import: try gpu first, fallback to cpu ----
let ort: any;
try {
  ort = await import("onnxruntime-node-gpu");
  console.log("[ORT] using onnxruntime-node-gpu (v1.14, CUDA EP)");
} catch (e: any) {
  ort = await import("onnxruntime-node");
  console.log("[ORT] using onnxruntime-node (v1.26, CPU EP + try ROCm/WebGPU via LD)");
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const API_ROOT = join(__dirname, "..", "..", "..", "api");
const MODEL_PATH = join(API_ROOT, "..", "..", "data", "models", "sherpa-whisper-turbo");
const RESULTS_DIR = join(__dirname, "results");

const VOCAB_SIZE = 51866;
const DECODER_START_TOKEN = 50258;
const EOS_TOKEN = 50257;
const SR = 16000;
const N_FFT = 400;
const HOP = 160;
const N_MEL = 128;
const MAX_SEQ = 448;
const D_MODEL = 1280;
const N_LAYERS = 4;

// ---- Mel spectrogram ----
function hannWindow(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

function melFilterbank() {
  const nFft = N_FFT;
  const fMin = 0,
    fMax = SR / 2;
  const melMin = 2595 * Math.log10(1 + fMin / 700);
  const melMax = 2595 * Math.log10(1 + fMax / 700);
  const pts = new Float64Array(N_MEL + 2);
  for (let i = 0; i < N_MEL + 2; i++) pts[i] = melMin + ((melMax - melMin) * i) / (N_MEL + 1);
  const hz = [...pts].map((m) => 700 * (10 ** (m / 2595) - 1));
  const bins = hz.map((h) => Math.floor(((nFft + 1) * h) / SR));
  const banks: Float64Array[] = [];
  for (let i = 0; i < N_MEL; i++) {
    const bank = new Float64Array(nFft / 2 + 1);
    const l = bins[i],
      c = bins[i + 1],
      r = bins[i + 2];
    if (c > l) for (let j = l; j < c; j++) bank[j] = (j - l) / (c - l);
    if (r > c) for (let j = c; j <= r; j++) bank[j] = (r - j) / (r - c);
    banks.push(bank);
  }
  return banks;
}

const BANKS = melFilterbank();
const WIN = hannWindow(N_FFT);

function dftMag2(frame: Float64Array): Float64Array {
  const mag2 = new Float64Array(N_FFT / 2 + 1);
  for (let k = 0; k <= N_FFT / 2; k++) {
    let re = 0,
      im = 0;
    for (let i = 0; i < N_FFT; i++) {
      const a = (-2 * Math.PI * k * i) / N_FFT;
      re += frame[i] * Math.cos(a);
      im += frame[i] * Math.sin(a);
    }
    mag2[k] = re * re + im * im;
  }
  return mag2;
}

function computeLogMel(pcm: Float32Array): Float32Array {
  const nFrames = Math.max(Math.floor((pcm.length - N_FFT + HOP) / HOP), 0);
  const mel = new Float64Array(N_MEL * nFrames);
  for (let i = 0; i < nFrames; i++) {
    const start = i * HOP;
    const frame = new Float64Array(N_FFT);
    for (let j = 0; j < N_FFT; j++)
      frame[j] = (start + j < pcm.length ? pcm[start + j] : 0) * WIN[j];
    const mag2 = dftMag2(frame);
    for (let m = 0; m < N_MEL; m++) {
      let sum = 0;
      for (let k = 0; k < mag2.length; k++) sum += mag2[k] * BANKS[m][k];
      mel[i * N_MEL + m] = Math.log10(Math.max(sum, 1e-10));
    }
  }
  for (let m = 0; m < N_MEL; m++) {
    let sum = 0,
      sq = 0;
    for (let i = 0; i < nFrames; i++) {
      const v = mel[i * N_MEL + m];
      sum += v;
      sq += v * v;
    }
    const mean = sum / nFrames;
    const std = Math.sqrt(Math.max(sq / nFrames - mean * mean, 1e-10));
    for (let i = 0; i < nFrames; i++) mel[i * N_MEL + m] = (mel[i * N_MEL + m] - mean) / std;
  }
  const out = new Float32Array(N_MEL * 3000);
  for (let m = 0; m < N_MEL; m++)
    for (let i = 0; i < nFrames; i++) out[m * 3000 + i] = mel[i * N_MEL + m];
  return out;
}

function loadAudio(filePath: string): Float32Array {
  const ffmpegBin = process.env.FFMPEG_PATH || "ffmpeg";
  const r = spawnSync(
    ffmpegBin,
    ["-i", filePath, "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1", "-"],
    { stdio: ["pipe", "pipe", "pipe"] },
  );
  if (r.error)
    throw new Error(
      `ffmpeg not found (FFMPEG_PATH="${ffmpegBin}"). Install ffmpeg or set FFMPEG_PATH.\n${r.error.message}`,
    );
  if (r.status !== 0) throw new Error(`ffmpeg exit ${r.status}`);
  return new Float32Array(r.stdout.buffer);
}

function loadTokens(): string[] {
  return readFileSync(join(MODEL_PATH, "turbo-tokens.txt"), "utf-8").trim().split("\n");
}

function tokensToText(tokens: number[], vocab: string[]): string {
  const bytes: number[] = [];
  for (const id of tokens) {
    const t = vocab[id];
    if (!t || t.includes("<|") || t.includes("|>")) continue;
    bytes.push(...Buffer.from(t, "base64"));
  }
  return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes)).trim();
}

// ---- Benchmark ----
interface Result {
  ep: string;
  ep_available: boolean;
  load_time_s: number;
  encode_time_ms: number;
  decode_time_ms: number;
  total_s: number;
  audio_s: number;
  rtf: number;
  tokens: number;
  text: string;
  error?: string;
}

async function runSingle(ort: any, ep: string[]): Promise<Result> {
  const label = ep.join("+");
  const encPath = join(MODEL_PATH, "turbo-encoder.int8.onnx");
  const decPath = join(MODEL_PATH, "turbo-decoder.int8.onnx");

  const t0 = performance.now();
  let enc: any, dec: any;
  try {
    enc = await ort.InferenceSession.create(encPath, { executionProviders: ep });
    console.log(`  enc OK`);
  } catch (e: any) {
    console.log(`  enc FAIL: ${e.message?.slice(0, 80)}`);
    return {
      ep: label,
      ep_available: false,
      load_time_s: (performance.now() - t0) / 1000,
      encode_time_ms: 0,
      decode_time_ms: 0,
      total_s: 0,
      audio_s: 0,
      rtf: Infinity,
      tokens: 0,
      text: "",
      error: `enc: ${e.message?.slice(0, 100)}`,
    };
  }
  try {
    dec = await ort.InferenceSession.create(decPath, { executionProviders: ep });
    console.log(`  dec OK`);
  } catch (e: any) {
    console.log(`  dec FAIL: ${e.message?.slice(0, 80)}`);
    await enc.release();
    return {
      ep: label,
      ep_available: false,
      load_time_s: (performance.now() - t0) / 1000,
      encode_time_ms: 0,
      decode_time_ms: 0,
      total_s: 0,
      audio_s: 0,
      rtf: Infinity,
      tokens: 0,
      text: "",
      error: `dec: ${e.message?.slice(0, 100)}`,
    };
  }
  const loadTime = (performance.now() - t0) / 1000;

  try {
    const pcm = loadAudio("/tmp/chirp.wav");
    const audioS = pcm.length / SR;
    const mel = computeLogMel(pcm);
    const melTensor = new ort.Tensor("float32", mel, [1, N_MEL, 3000]);

    const te1 = performance.now();
    const { n_layer_cross_k, n_layer_cross_v } = await enc.run({ mel: melTensor });
    const encodeMs = performance.now() - te1;

    let selfK = new ort.Tensor("float32", new Float32Array(N_LAYERS * MAX_SEQ * D_MODEL), [
      N_LAYERS,
      1,
      MAX_SEQ,
      D_MODEL,
    ]);
    let selfV = new ort.Tensor("float32", new Float32Array(N_LAYERS * MAX_SEQ * D_MODEL), [
      N_LAYERS,
      1,
      MAX_SEQ,
      D_MODEL,
    ]);
    const tk: number[] = [];

    const td1 = performance.now();
    for (let step = 0; step < MAX_SEQ; step++) {
      const tokenId = step === 0 ? DECODER_START_TOKEN : tk[step - 1];
      const feeds: Record<string, any> = {
        tokens: new ort.Tensor("int64", BigInt64Array.from([BigInt(tokenId)]), [1, 1]),
        in_n_layer_self_k_cache: selfK,
        in_n_layer_self_v_cache: selfV,
        n_layer_cross_k,
        n_layer_cross_v,
        offset: new ort.Tensor("int64", BigInt64Array.from([BigInt(step)]), [1]),
      };
      const o = await dec.run(feeds);
      const logits = (o.logits.data as Float32Array).slice(-VOCAB_SIZE);
      let mx = -Infinity,
        nt = 0;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > mx) {
          mx = logits[i];
          nt = i;
        }
      }
      if (nt === EOS_TOKEN) break;
      tk.push(nt);
      selfK = o.out_n_layer_self_k_cache;
      selfV = o.out_n_layer_self_v_cache;
    }
    const decodeMs = performance.now() - td1;

    const totalS = (performance.now() - t0) / 1000;
    const vocab = loadTokens();
    const text = tokensToText(tk, vocab);

    await enc.release();
    await dec.release();

    return {
      ep: label,
      ep_available: true,
      load_time_s: loadTime,
      encode_time_ms: encodeMs,
      decode_time_ms: decodeMs,
      total_s: totalS,
      audio_s: audioS,
      rtf: totalS / audioS,
      tokens: tk.length,
      text,
    };
  } catch (e: any) {
    await enc.release();
    await dec.release();
    return {
      ep: label,
      ep_available: true,
      load_time_s: loadTime,
      encode_time_ms: 0,
      decode_time_ms: 0,
      total_s: (performance.now() - t0) / 1000,
      audio_s: 0,
      rtf: Infinity,
      tokens: 0,
      text: "",
      error: e.message?.slice(0, 200),
    };
  }
}

async function runOne(ep: string[], suffix: string) {
  console.log(`\n--- ${ep.join("+")} ---`);
  const result = await runSingle(ort, ep);
  mkdirSync(RESULTS_DIR, { recursive: true });
  const out = join(RESULTS_DIR, `ts-onnx-${suffix}.json`);
  writeFileSync(out, JSON.stringify(result, null, 2));
  console.log(`Saved: ${out}`);
  return result;
}

async function main() {
  console.log("=== Whisper ONNX GPU Benchmark ===\n");

  const allResults: Record<string, Result> = {};

  // webgpu (no CUDA, no ROCm — just test if available)
  allResults.webgpu = await runOne(["webgpu", "cpu"], "webgpu");

  // rocm via onnxruntime-node (likely CPU fallback)
  allResults.rocm = await runOne(["rocm", "cpu"], "rocm");

  // cpu baseline
  allResults.cpu = await runOne(["cpu"], "cpu");

  console.log("\n=== Summary ===");
  for (const [k, r] of Object.entries(allResults)) {
    if (!r.ep_available) console.log(`  ts-onnx-${k.padEnd(10)} UNAVAILABLE  ${r.error}`);
    else if (r.error) console.log(`  ts-onnx-${k.padEnd(10)} ERROR  ${r.error}`);
    else
      console.log(
        `  ts-onnx-${k.padEnd(10)} RTF=${r.rtf.toFixed(2)}  ${r.tokens} tokens  "${r.text.slice(0, 50)}"`,
      );
  }
}

main().catch(console.error);
