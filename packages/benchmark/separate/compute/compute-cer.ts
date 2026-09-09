import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve, basename, extname } from 'node:path';

const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..');
const PYTHON_BIN = join(REPO_ROOT, '.venv', 'bin', 'python');
const WER_PY = resolve(__dirname, 'wer.py');
const GROUND_TRUTH = resolve(REPO_ROOT, 'packages', 'benchmark', 'ref', 'metadata', 'srt_manual.json');

function usage(): never {
  console.error(`
Usage: bun run compute/compute-cer.ts <path> [--label name]

<path>  can be:
  - a results directory containing metadata/asr.json
  - a direct path to an asr.json file

Options:
  --label name   Label for output file (default: derived from dir/file name)
`);
  process.exit(1);
}

function parseArgs(): { inputPath: string; label: string } {
  const args = process.argv.slice(2);
  if (args.length === 0) usage();
  let inputPath = '';
  let label = '';
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--label') {
      label = args[++i] || '';
    } else {
      inputPath = args[i];
    }
  }
  if (!inputPath) usage();
  return { inputPath: resolve(inputPath), label };
}

function resolveAsrPath(inputPath: string): string {
  if (existsSync(inputPath) && !inputPath.endsWith('.json')) {
    const candidate = join(inputPath, 'metadata', 'asr.json');
    if (existsSync(candidate)) return candidate;
    const candidate2 = join(inputPath, 'asr.json');
    if (existsSync(candidate2)) return candidate2;
    throw new Error(`Directory ${inputPath} contains neither metadata/asr.json nor asr.json`);
  }
  if (!existsSync(inputPath)) {
    throw new Error(`File not found: ${inputPath}`);
  }
  return inputPath;
}

interface Segment { text: string; start: number; end: number }
interface Normalized { text: string; segments: Segment[] }

function normalizeTranscription(data: any): Normalized {
  // Format 1: { result: { text, segments } } (standard)
  if (data?.result?.text != null && data?.result?.segments) {
    return { text: data.result.text, segments: data.result.segments };
  }
  // Format 2: { transcription: [...] } (pipeline whisper.cpp JSON)
  if (Array.isArray(data?.transcription)) {
    const segments: Segment[] = data.transcription.map((s: any) => ({
      text: (s.text || '').trim(),
      start: s.offsets?.from ?? 0,
      end: s.offsets?.to ?? 0,
    }));
    const text = segments.map(s => s.text).join(' ');
    return { text, segments };
  }
  throw new Error('Unrecognized ASR JSON format: missing result.text or transcription array');
}

function computeWER(hypFile: string): any {
  const r = spawnSync(PYTHON_BIN, [WER_PY, GROUND_TRUTH, hypFile], {
    timeout: 30_000,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  if (r.status !== 0) {
    throw new Error(`wer.py failed: ${r.stderr?.toString().slice(-300)}`);
  }
  return JSON.parse(r.stdout.toString());
}

function main() {
  const { inputPath, label: labelArg } = parseArgs();
  const asrPath = resolveAsrPath(inputPath);
  const metadataDir = resolve(asrPath, '..');
  const resultsDir = resolve(metadataDir, '..');
  const defaultLabel = basename(resultsDir).replace(/^results-/, '').replace(/^ggml-/, '');
  const label = labelArg || defaultLabel;

  console.log(`[compute-cer] Reading: ${asrPath}`);
  const raw = JSON.parse(readFileSync(asrPath, 'utf-8'));
  const { text, segments } = normalizeTranscription(raw);
  console.log(`[compute-cer] Segments: ${segments.length}, chars: ${text.length}`);

  const hypFile = join(metadataDir, `wer-${label}.json`);
  writeFileSync(hypFile, JSON.stringify({
    audio_info: { duration: segments.length ? segments[segments.length - 1].end : 0 },
    result: { text, segments },
    _source: asrPath,
    _label: label,
  }, null, 2));

  const result = computeWER(hypFile);
  const summaryFile = join(metadataDir, `wer-${label}-summary.json`);
  const summary = { label, ...result, _source: asrPath };
  writeFileSync(summaryFile, JSON.stringify(summary, null, 2));

  console.log(`\n=== WER/CER: ${label} ===`);
  console.log(`  Ref words: ${result.ref_words}, Hyp words: ${result.hyp_words}`);
  console.log(`  Ref chars: ${result.ref_chars}, Hyp chars: ${result.hyp_chars}`);
  console.log(`  WER:  ${(result.wer * 100).toFixed(2)}% (subs=${result.word_subs}, ins=${result.word_ins}, del=${result.word_dels})`);
  console.log(`  CER:  ${(result.cer * 100).toFixed(2)}% (subs=${result.char_subs}, ins=${result.char_ins}, del=${result.char_dels})`);
  console.log(`\nSaved: ${hypFile}`);
  console.log(`Saved: ${summaryFile}`);
}

main();
