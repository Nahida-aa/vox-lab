import { spawnSync } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { RESULTS_DIR, VIDEO_PATH, REPO_ROOT, round, type BenchmarkResult } from "./bench-shared";

const WHISPER_CLI = join(REPO_ROOT, "submodule", "whisper.cpp", "build", "bin", "whisper-cli");
const MODEL = join(REPO_ROOT, "data", "models", "whisper", "ggml-large-v3-turbo.bin");

function parseTiming(output: string, key: string): number {
  const m = output.match(new RegExp(`whisper_print_timings:\\s+${key}\\s+=\\s+([\\d.]+)\\s+ms`));
  return m ? parseFloat(m[1]) / 1000 : 0;
}

function measureLoadTime(noGpu: boolean): number {
  const silentWav = join(REPO_ROOT, "packages", "tmp", "_whisper_cli_silent.wav");
  if (
    spawnSync(
      "ffmpeg",
      [
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-t",
        "1",
        "-c:a",
        "pcm_s16le",
        silentWav,
      ],
      { timeout: 10_000 },
    ).status !== 0
  ) {
    return 0;
  }
  const args = ["-m", MODEL, silentWav, "-l", "en", "-t", "4"];
  if (noGpu) args.push("-ng");
  const result = spawnSync(WHISPER_CLI, args, {
    timeout: 120_000,
    env: { ...process.env, GGML_VK_DISABLE_DOT2: "1" },
  });
  return parseTiming(result.stderr?.toString() || "", "load time");
}

export async function benchmarkWhisperCli(): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  const configs = [
    { label: "GPU (Vulkan)", device: "gpu", noGpu: false },
    { label: "CPU", device: "cpu", noGpu: true },
  ];

  for (const cfg of configs) {
    console.log(`[whisper-cli] ${cfg.label} — measuring load time...`);
    const loadTimeS = round(measureLoadTime(cfg.noGpu), 3);
    console.log(`  Load: ${loadTimeS}s`);

    console.log(`[whisper-cli] ${cfg.label} — transcribing...`);
    const outDir = join(RESULTS_DIR, `wcli-${cfg.device}-${Date.now()}`);
    mkdirSync(outDir, { recursive: true });

    const wavPath = join(outDir, "audio.wav");
    const ffmpegR = spawnSync(
      "ffmpeg",
      ["-y", "-i", VIDEO_PATH, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wavPath],
      { timeout: 120_000 },
    );
    if (ffmpegR.status !== 0) {
      console.error(`  ffmpeg failed: ${ffmpegR.stderr?.toString().slice(-200)}`);
      rmSync(outDir, { recursive: true, force: true });
      continue;
    }

    const whisperArgs = ["-m", MODEL, wavPath, "-l", "en", "-t", "4"];
    if (cfg.noGpu) whisperArgs.push("-ng");

    const t1 = performance.now();
    const result = spawnSync(WHISPER_CLI, whisperArgs, {
      timeout: 600_000,
      env: { ...process.env, GGML_VK_DISABLE_DOT2: "1" },
    });
    const elapsed = (performance.now() - t1) / 1000;

    if (result.status !== 0) {
      console.error(`  whisper-cli failed: ${result.stderr?.toString().slice(-200)}`);
      rmSync(outDir, { recursive: true, force: true });
      continue;
    }

    const output = result.stderr?.toString() || result.stdout?.toString() || "";
    const totalTimeS = parseTiming(output, "total time");
    const processTimeS = round(totalTimeS > 0 ? totalTimeS - loadTimeS : elapsed, 3);

    results.push({
      engine: "whisper-cli",
      device: cfg.device,
      computeType: "gguf-float16",
      audioDurationS: 170,
      loadTimeS,
      processTimeS,
      totalTimeS: round(loadTimeS + processTimeS, 3),
      rtf: round(processTimeS / 170, 3),
    });

    console.log(`  RTF: ${results[results.length - 1].rtf}`);

    rmSync(outDir, { recursive: true, force: true });
  }

  return results;
}
