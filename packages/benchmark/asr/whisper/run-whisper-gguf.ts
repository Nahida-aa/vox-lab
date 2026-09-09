import { spawnSync } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { cpus, homedir } from "node:os";
import {
  RESULTS_DIR,
  VIDEO_PATH,
  REPO_ROOT,
  PYTHON_BIN,
  round,
  type BenchmarkResult,
} from "./bench-shared";

const SCRIPT = join(REPO_ROOT, "packages", "cli", "scripts", "asr", "pywhispercpp_bench.py");
const GGUF_MODEL = join(REPO_ROOT, "data", "models", "whisper", "ggml-large-v3-turbo.bin");
const N_THREADS = cpus().length;

function measureLoadTime(): number {
  const t0 = performance.now();
  const result = spawnSync(
    PYTHON_BIN,
    [SCRIPT, "--benchmark-load", "--model", GGUF_MODEL, "--n-threads", String(N_THREADS)],
    { timeout: 300_000 },
  );
  if (result.status !== 0) {
    throw new Error(`--benchmark-load failed: ${result.stderr?.toString().slice(-300)}`);
  }
  return (performance.now() - t0) / 1000;
}

export async function benchmarkWhisperGGUF(): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  console.log("[whisper.cpp/GGUF] Measuring load time...");

  let loadTimeS: number;
  try {
    loadTimeS = round(measureLoadTime(), 3);
    console.log(`  Load: ${loadTimeS}s`);
  } catch (err) {
    console.error(`  Load failed: ${err}`);
    return results;
  }

  console.log("[whisper.cpp/GGUF] Transcribing...");
  const outDir = join(RESULTS_DIR, `gguf-${Date.now()}`);
  mkdirSync(outDir, { recursive: true });

  const t1 = performance.now();
  const procArgs = [
    SCRIPT,
    VIDEO_PATH,
    outDir,
    "en",
    "--model",
    GGUF_MODEL,
    "--n-threads",
    String(N_THREADS),
  ];
  const result = spawnSync(PYTHON_BIN, procArgs, { timeout: 600_000 });

  if (result.status !== 0) {
    console.error(`  Transcribe failed: ${result.stderr?.toString().slice(-300)}`);
    rmSync(outDir, { recursive: true, force: true });
    return results;
  }

  const processTimeS = round((performance.now() - t1) / 1000, 3);

  results.push({
    engine: "whisper.cpp",
    device: "cpu",
    computeType: "gguf-float16",
    audioDurationS: 170,
    loadTimeS,
    processTimeS,
    totalTimeS: round(loadTimeS + processTimeS, 3),
    rtf: round(processTimeS / 170, 3),
  });

  console.log(`  RTF: ${results[0].rtf}`);

  rmSync(outDir, { recursive: true, force: true });
  return results;
}
