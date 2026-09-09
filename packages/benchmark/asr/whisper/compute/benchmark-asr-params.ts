import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  copyFileSync,
  unlinkSync,
} from "node:fs";
import { join, resolve, extname } from "node:path";

const REPO_ROOT = resolve(__dirname, "..", "..", "..", "..", "..");
const WHISPER_CLI = join(REPO_ROOT, "submodule", "whisper.cpp", "build", "bin", "whisper-vulkan");
const MODEL =
  process.env.WHISPER_MODEL ||
  join(REPO_ROOT, "data", "models", "whisper", "ggml-large-v3-turbo.bin");
const PYTHON_BIN = join(REPO_ROOT, ".venv", "bin", "python");
const VAD_MODEL = join(REPO_ROOT, "data", "models", "whisper", "ggml-silero-v5.1.2.bin");
const VAD_MODEL_V6 = join(REPO_ROOT, "data", "models", "whisper", "ggml-silero-v6.2.0.bin");
const WER_PY = resolve(__dirname, "..", "..", "..", "separate", "compute", "wer.py");
const GROUND_TRUTH = resolve(
  REPO_ROOT,
  "packages",
  "benchmark",
  "ref",
  "metadata",
  "asr_manual.json",
);
const RESULTS_BASE = resolve(__dirname, "..", "results");
const TMP = resolve(REPO_ROOT, "packages", "tmp", "benchmark-asr-params");

interface AudioSource {
  label: string;
  path: string;
}

interface ParamSet {
  label: string;
  args: string[];
}

const AUDIO_SOURCES: AudioSource[] = [
  {
    label: "ggml-s1-raw",
    path: resolve(
      REPO_ROOT,
      "packages",
      "benchmark",
      "separate",
      "ref",
      "media",
      "video_source.mp4",
    ),
  },
  {
    label: "ggml-s1-sidechain",
    path: resolve(
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
  },
  {
    label: "ggml-s1-vocals",
    path: resolve(
      REPO_ROOT,
      "packages",
      "benchmark",
      "separate",
      "results",
      "ggml-s1",
      "media",
      "target_3_vocals.wav",
    ),
  },
];

const PARAM_SETS: ParamSet[] = [
  { label: "baseline", args: [] },
  { label: "vad-default", args: ["--vad"] },
  { label: "vad-low", args: ["--vad", "--vad-threshold", "0.3"] },
  { label: "vad-high", args: ["--vad", "--vad-threshold", "0.7"] },
  { label: "nst-050", args: ["--no-speech-thold", "0.50"] },
  { label: "nst-040", args: ["--no-speech-thold", "0.40"] },
  { label: "vad+nst-050", args: ["--vad", "--no-speech-thold", "0.50"] },
  { label: "temp-02", args: ["--temperature", "0.2"] },
  { label: "temp-04", args: ["--temperature", "0.4"] },
  // Round 2: short/quiet speech detection
  { label: "nst-020", args: ["--no-speech-thold", "0.20"] },
  { label: "nst-010", args: ["--no-speech-thold", "0.10"] },
  { label: "max-len-20", args: ["--max-len", "20", "--split-on-word"] },
  { label: "suppress-nst", args: ["--suppress-nst"] },
  {
    label: "nst-020+max20",
    args: ["--no-speech-thold", "0.20", "--max-len", "20", "--split-on-word"],
  },
  { label: "vad+nst-030", args: ["--vad", "--no-speech-thold", "0.30"] },
  { label: "vad-v6", args: ["--vad", "-vm", VAD_MODEL_V6] },
  { label: "vad-v6+nst-020", args: ["--vad", "-vm", VAD_MODEL_V6, "--no-speech-thold", "0.20"] },
  { label: "vad-v6-th02", args: ["--vad", "-vm", VAD_MODEL_V6, "--vad-threshold", "0.2"] },
  // Round 3: prompt engineering
  { label: "prompt", args: ["--prompt", "转录所有声音，包括语气词如嗯啊唉哈哈"] },
  {
    label: "prompt+carry",
    args: ["--prompt", "转录所有声音，包括语气词如嗯啊唉哈哈", "--carry-initial-prompt"],
  },
];

function ensureWav(source: AudioSource): string {
  const ext = extname(source.path).toLowerCase();
  if (ext === ".wav") return source.path;
  // Convert mp4/etc to wav in tmp
  mkdirSync(TMP, { recursive: true });
  const out = join(TMP, `${source.label}.wav`);
  if (existsSync(out)) return out;
  console.log(`  converting ${ext} to wav...`);
  const r = spawnSync(
    "ffmpeg",
    ["-y", "-i", source.path, "-ac", "1", "-ar", "16000", "-f", "wav", out],
    { timeout: 120_000 },
  );
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${r.stderr?.toString().slice(-200)}`);
  return out;
}

function whisperTranscribe(
  audioPath: string,
  extraArgs: string[],
): { raw: any; elapsedMs: number; timing: { totalMs: number; loadMs: number } } {
  const jsonPath = `${audioPath}.json`;
  if (existsSync(jsonPath)) unlinkSync(jsonPath); // clean previous

  const args = ["-m", MODEL, audioPath, "-l", "zh", "-t", "4", "-ojf"];
  if (extraArgs.includes("--vad")) args.push("-vm", VAD_MODEL);
  args.push(...extraArgs);
  const t0 = performance.now();
  const r = spawnSync(WHISPER_CLI, args, { timeout: 600_000 });
  const elapsedMs = performance.now() - t0;

  // Parse timing from stderr
  const stderr = r.stderr?.toString() || "";
  const tm = stderr.match(/whisper_print_timings:\s+total time\s+=\s+([\d.]+)\s+ms/);
  const lm = stderr.match(/whisper_print_timings:\s+load time\s+=\s+([\d.]+)\s+ms/);
  const timing = {
    totalMs: tm ? parseFloat(tm[1]) : 0,
    loadMs: lm ? parseFloat(lm[1]) : 0,
  };

  if (r.status !== 0 && r.status !== null) {
    throw new Error(`whisper-vulkan exit ${r.status}: ${stderr.slice(-300)}`);
  }
  if (!existsSync(jsonPath))
    throw new Error(`whisper did not produce ${jsonPath}: ${stderr.slice(-300)}`);

  const raw = JSON.parse(readFileSync(jsonPath, "utf-8"));
  unlinkSync(jsonPath); // clean up
  return { raw, elapsedMs, timing };
}

interface Segment {
  text: string;
  start: number;
  end: number;
  confidence?: { avg: number; min: number };
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

function computeCER(asrPath: string): any {
  const r = spawnSync(PYTHON_BIN, [WER_PY, GROUND_TRUTH, asrPath], {
    timeout: 30_000,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
  });
  if (r.status !== 0) throw new Error(`wer.py failed: ${r.stderr?.toString().slice(-300)}`);
  return JSON.parse(r.stdout.toString());
}

function runOne(source: AudioSource, param: ParamSet) {
  const label = `${source.label}_${param.label}`;
  const outDir = join(RESULTS_BASE, source.label, param.label);
  const metadataDir = join(outDir, "metadata");
  mkdirSync(metadataDir, { recursive: true });

  console.log(`\n=== ${label} ===`);
  console.log(`  audio: ${source.path}`);
  console.log(`  args: ${param.args.join(" ") || "(none)"}`);

  const audioWav = ensureWav(source);

  const { raw, elapsedMs, timing } = whisperTranscribe(audioWav, param.args);

  const segments = whisperToSegments(raw);
  const text = segments.map((s) => s.text).join(" ");
  const audioDurMs = segments.length > 0 ? segments[segments.length - 1].end : 0;
  const rtf = audioDurMs > 0 ? timing.totalMs / audioDurMs : 0;

  writeFileSync(join(metadataDir, "whisper_raw.json"), JSON.stringify(raw, null, 2));

  const asrOutput = {
    audio_info: { duration: audioDurMs },
    result: { text, segments },
    _engine: "whisper.cpp",
    _device: "vulkan",
    _rtf: rtf.toFixed(3),
    _params: { source: source.label, ...param },
  };
  const asrPath = join(metadataDir, "asr.json");
  writeFileSync(asrPath, JSON.stringify(asrOutput, null, 2));

  const werResult = computeCER(asrPath);

  const summary = {
    label,
    source: source.label,
    params: param,
    segments: segments.length,
    audio_duration_s: parseFloat((audioDurMs / 1000).toFixed(1)),
    whisper_total_ms: timing.totalMs,
    whisper_load_ms: timing.loadMs,
    elapsed_ms: elapsedMs,
    rtf: parseFloat(rtf.toFixed(3)),
    wer: werResult.wer,
    cer: werResult.cer,
    hyp_chars: werResult.hyp_chars,
    ref_chars: werResult.ref_chars,
    hyp_words: werResult.hyp_words,
    ref_words: werResult.ref_words,
  };
  writeFileSync(join(metadataDir, "summary.json"), JSON.stringify(summary, null, 2));

  console.log(
    `  segs=${segments.length} dur=${(audioDurMs / 1000).toFixed(1)}s RTF=${rtf.toFixed(3)}`,
  );
  console.log(
    `  WER=${(werResult.wer * 100).toFixed(2)}% CER=${(werResult.cer * 100).toFixed(2)}%`,
  );

  return summary;
}

// --- Main ---
if (require.main === module) {
  const results: any[] = [];

  for (const source of AUDIO_SOURCES) {
    for (const param of PARAM_SETS) {
      const summaryPath = join(RESULTS_BASE, source.label, param.label, "metadata", "summary.json");
      if (existsSync(summaryPath)) {
        console.log(`[skip] ${source.label}_${param.label} already done`);
        results.push(JSON.parse(readFileSync(summaryPath, "utf-8")));
        continue;
      }
      results.push(runOne(source, param));
    }
  }

  console.log("\n======= SUMMARY =======");
  console.log(
    "source          | param          | segs | dur(s)  | RTF    | WER%   | CER%   | hyp_ch | ref_ch",
  );
  console.log(
    "----------------|----------------|------|---------|--------|--------|--------|-------|-------",
  );
  for (const r of results) {
    const src = r.source.padEnd(16);
    const param = r.params.label.padEnd(16);
    const segs = String(r.segments).padStart(4);
    const dur = String(r.audio_duration_s).padStart(7);
    const rtf = String(r.rtf).padStart(6);
    const wer = (r.wer * 100).toFixed(2).padStart(6);
    const cer = (r.cer * 100).toFixed(2).padStart(6);
    const hc = String(r.hyp_chars).padStart(7);
    const rc = String(r.ref_chars).padStart(7);
    console.log(
      `${src} | ${param} | ${segs} | ${dur} | ${rtf} | ${wer}% | ${cer}% | ${hc} | ${rc}`,
    );
  }
}
