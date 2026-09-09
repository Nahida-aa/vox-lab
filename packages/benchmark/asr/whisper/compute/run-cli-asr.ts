// Run CLI-equivalent ASR on benchmark audio, save to benchmark results dir.
// Usage: bun run run-cli-asr.ts <source-label> <param-label> [--vad-threshold 0.2]
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync, readFileSync, unlinkSync } from "node:fs";
import { join, resolve } from "node:path";

const REPO_ROOT = resolve(__dirname, "..", "..", "..", "..", "..");
const WHISPER_CLI = join(REPO_ROOT, "submodule", "whisper.cpp", "build", "bin", "whisper-vulkan");
const MODEL =
  process.env.WHISPER_MODEL ||
  join(REPO_ROOT, "data", "models", "whisper", "ggml-large-v3-turbo.bin");
const VAD_MODEL = join(REPO_ROOT, "data", "models", "whisper", "ggml-silero-v5.1.2.bin");
const VAD_MODEL_V6 = join(REPO_ROOT, "data", "models", "whisper", "ggml-silero-v6.2.0.bin");
const RESULTS_BASE = resolve(__dirname, "..", "results");
const GROUND_TRUTH = resolve(
  REPO_ROOT,
  "packages",
  "benchmark",
  "ref",
  "metadata",
  "asr_manual.json",
);

interface Segment {
  text: string;
  start: number;
  end: number;
  confidence?: { avg: number; min: number };
}

function srtTime(ms: number): string {
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const ml = ms % 1000;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ml).padStart(3, "0")}`;
}

function whisperToSegments(raw: any): Segment[] {
  return (raw.transcription || []).map((s: any) => {
    const tokens = (s.tokens || []).filter((t: any) => {
      const txt = t.text?.trim();
      return txt && !txt.startsWith("[") && t.p != null;
    });
    const probs = tokens.map((t: any) => t.p);
    const confidence =
      probs.length > 0
        ? {
            avg: +(probs.reduce((a: number, b: number) => a + b, 0) / probs.length).toFixed(4),
            min: +Math.min(...probs).toFixed(4),
          }
        : undefined;
    return {
      text: (s.text || "").trim(),
      start: s.offsets?.from ?? 0,
      end: s.offsets?.to ?? 0,
      confidence,
    };
  });
}

const AUDIO_SOURCES: Record<string, string> = {
  "ggml-s1-raw": resolve(
    REPO_ROOT,
    "packages",
    "benchmark",
    "separate",
    "ref",
    "media",
    "video_source.mp4",
  ),
  "ggml-s1-sidechain": resolve(
    REPO_ROOT,
    "packages",
    "benchmark",
    "separate",
    "results",
    "ggml-s1-sidechain",
    "sc_t0.1_r20_a1_rel200_bgm-12",
    "media",
    "target_3_vocals_mixed.wav",
  ),
  "ggml-s1-vocals": resolve(
    REPO_ROOT,
    "packages",
    "benchmark",
    "separate",
    "results",
    "ggml-s1",
    "media",
    "target_3_vocals.wav",
  ),
};

function main() {
  const sourceLabel = process.argv[2];
  const paramLabel = process.argv[3];
  if (!sourceLabel || !paramLabel) {
    console.error(
      "Usage: bun run run-cli-asr.ts <source-label> <param-label> [extra whisper args...]",
    );
    console.error("Sources:", Object.keys(AUDIO_SOURCES).join(", "));
    process.exit(1);
  }

  const sourcePath = AUDIO_SOURCES[sourceLabel];
  if (!sourcePath) throw new Error(`Unknown source: ${sourceLabel}`);

  const extraArgs = process.argv.slice(4);
  const lang = "zh";

  // 1. Preprocess audio — same as CLI asrWhisperCpp: copy .wav as-is, convert others to mono
  const tmpAudio = join(REPO_ROOT, "packages", "tmp", `cli-asr-${sourceLabel}-${paramLabel}.wav`);
  if (sourcePath.endsWith(".wav")) {
    spawnSync("cp", [sourcePath, tmpAudio], { timeout: 10_000 });
  } else {
    spawnSync("ffmpeg", ["-y", "-i", sourcePath, "-ac", "1", tmpAudio], { timeout: 30_000 });
  }

  // 2. Build whisper args — same pattern as CLI's asrWhisperCpp
  const whisperArgs = ["-m", MODEL, tmpAudio, "-l", lang, "-t", "4", "-ojf"];
  if (extraArgs.includes("--vad")) {
    const vmIdx = extraArgs.indexOf("-vm");
    const vadModel = vmIdx !== -1 ? extraArgs[vmIdx + 1] : VAD_MODEL_V6;
    whisperArgs.push("--vad", "-vm", vadModel);
  }
  // Add remaining extra args that aren't --vad/-vm
  for (let i = 0; i < extraArgs.length; i++) {
    if (extraArgs[i] === "--vad") continue;
    if (extraArgs[i] === "-vm") {
      i++;
      continue;
    }
    whisperArgs.push(extraArgs[i]);
  }

  console.log(`[CLI-ASR] source=${sourceLabel} param=${paramLabel}`);
  console.log(
    `  audio: ${tmpAudio} (${((existsSync(tmpAudio) ? require("fs").statSync(tmpAudio).size : 0) / 1024 / 1024).toFixed(1)} MB)`,
  );
  console.log(`  whisper: ${WHISPER_CLI}`);
  console.log(`  args: ${whisperArgs.join(" ")}`);

  // 3. Run whisper
  const jsonPath = `${tmpAudio}.json`;
  if (existsSync(jsonPath)) unlinkSync(jsonPath);

  const t0 = performance.now();
  const r = spawnSync(WHISPER_CLI, whisperArgs, { timeout: 600_000 });
  const elapsedMs = performance.now() - t0;

  if (r.status !== 0 && r.status !== null) {
    throw new Error(`whisper exit ${r.status}: ${r.stderr?.toString().slice(-300)}`);
  }
  if (!existsSync(jsonPath)) throw new Error(`whisper did not produce ${jsonPath}`);

  const raw = JSON.parse(readFileSync(jsonPath, "utf-8"));
  unlinkSync(jsonPath);
  unlinkSync(tmpAudio);

  // 4. Parse segments — same shape as CLI output
  const segments = whisperToSegments(raw);
  const text = segments.map((s) => s.text).join(" ");
  const lastEndMs = segments.length ? segments[segments.length - 1].end : 0;

  const asrOutput = {
    audio_info: { duration: lastEndMs },
    result: { text, segments },
    _engine: "whisper.cpp",
    _model: MODEL,
    _params: Object.fromEntries(
      [...extraArgs]
        .filter((_, i, a) => {
          if (a[i] === "-vm") return false;
          const prev = a[i - 1];
          return prev !== "-vm";
        })
        .map((v) => [v.replace(/^-+/, ""), true]),
    ),
    _input_audio: sourcePath,
    _device: "vulkan",
    _rtf: elapsedMs > 0 && lastEndMs > 0 ? (elapsedMs / lastEndMs).toFixed(3) : "0",
  };

  // 5. Save to benchmark results dir
  const resultDir = join(RESULTS_BASE, sourceLabel, paramLabel);
  const metadataDir = join(resultDir, "metadata");
  mkdirSync(metadataDir, { recursive: true });

  writeFileSync(join(metadataDir, "asr.json"), JSON.stringify(asrOutput, null, 2));

  // Also save whisper_raw.json for reference
  writeFileSync(join(metadataDir, "whisper_raw.json"), JSON.stringify(raw, null, 2));

  // summary.json
  const stderr = r.stderr?.toString() || "";
  const tm = stderr.match(/whisper_print_timings:\s+total time\s+=\s+([\d.]+)\s+ms/);
  const lm = stderr.match(/whisper_print_timings:\s+load time\s+=\s+([\d.]+)\s+ms/);
  const summary = {
    label: `${sourceLabel}_${paramLabel}`,
    source: sourceLabel,
    params: { label: paramLabel, args: extraArgs },
    segments: segments.length,
    audio_duration_s: lastEndMs / 1000,
    whisper_total_ms: tm ? parseFloat(tm[1]) : 0,
    whisper_load_ms: lm ? parseFloat(lm[1]) : 0,
    elapsed_ms: elapsedMs,
    rtf: +(elapsedMs / (lastEndMs || 1)).toFixed(3),
    hyp_chars: text.length,
  };
  writeFileSync(join(metadataDir, "summary.json"), JSON.stringify(summary, null, 2));

  console.log(`[CLI-ASR] Done → ${resultDir}`);
  console.log(
    `  segments=${segments.length} chars=${text.length} elapsed=${(elapsedMs / 1000).toFixed(1)}s`,
  );
}

main();
