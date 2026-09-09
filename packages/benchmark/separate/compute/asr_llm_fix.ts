import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve, basename } from 'node:path';
import { segmentsToPrompt, parseLines, fixWithLLM, DEFAULT_API_BASE, DEFAULT_MODEL } from '../../../cli/src/feat/stages/asr/llm.ts';

const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..');
const PYTHON_BIN = join(REPO_ROOT, '.venv', 'bin', 'python');
const WER_PY = resolve(__dirname, 'wer.py');
const GROUND_TRUTH = resolve(REPO_ROOT, 'packages', 'benchmark', 'ref', 'metadata', 'srt_manual.json');

function usage(): never {
  console.error(`
Usage: bun run compute/asr_llm_fix.ts <path> [options]

<path>  can be a results directory or direct path to asr.json

Options:
  --label name         Label for output
  --model name         LLM model (default: ${DEFAULT_MODEL})
  --api-base url       LLM API base URL (default: ${DEFAULT_API_BASE})
  --domain-hint text   Domain hint for LLM context (e.g. "仙侠题材, 角色:叶白,慧天")
`);
  process.exit(1);
}

function parseArgs(): { inputPath: string; label: string; model: string; apiBase: string; domainHint: string } {
  const args = process.argv.slice(2);
  if (args.length === 0) usage();
  let inputPath = '', label = '', model = DEFAULT_MODEL, apiBase = DEFAULT_API_BASE, domainHint = '';
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--label') { label = args[++i] || ''; }
    else if (args[i] === '--model') { model = args[++i] || DEFAULT_MODEL; }
    else if (args[i] === '--api-base') { apiBase = args[++i] || DEFAULT_API_BASE; }
    else if (args[i] === '--domain-hint') { domainHint = args[++i] || ''; }
    else { inputPath = args[i]; }
  }
  if (!inputPath) usage();
  return { inputPath: resolve(inputPath), label, model, apiBase, domainHint };
}

function resolveAsrPath(inputPath: string): string {
  if (existsSync(inputPath) && !inputPath.endsWith('.json')) {
    const c = join(inputPath, 'metadata', 'asr.json');
    if (existsSync(c)) return c;
    const c2 = join(inputPath, 'asr.json');
    if (existsSync(c2)) return c2;
    throw new Error(`Directory ${inputPath} contains neither metadata/asr.json nor asr.json`);
  }
  if (!existsSync(inputPath)) throw new Error(`File not found: ${inputPath}`);
  return inputPath;
}

interface Segment { text: string; start: number; end: number }
interface Normalized { text: string; segments: Segment[] }

function normalizeTranscription(data: any): Normalized {
  if (data?.result?.text != null && data?.result?.segments) {
    return { text: data.result.text, segments: data.result.segments };
  }
  if (Array.isArray(data?.transcription)) {
    const segments: Segment[] = data.transcription.map((s: any) => ({
      text: (s.text || '').trim(),
      start: s.offsets?.from ?? 0,
      end: s.offsets?.to ?? 0,
    }));
    return { text: segments.map(s => s.text).join(' '), segments };
  }
  throw new Error('Unrecognized ASR JSON format');
}

function computeWER(refFile: string, hypFile: string): any {
  const r = spawnSync(PYTHON_BIN, [WER_PY, refFile, hypFile], {
    timeout: 30_000,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  if (r.status !== 0) throw new Error(`wer.py failed: ${r.stderr?.toString().slice(-300)}`);
  return JSON.parse(r.stdout.toString());
}

async function main() {
  const { inputPath, label: labelArg, model, apiBase, domainHint } = parseArgs();
  const asrPath = resolveAsrPath(inputPath);
  const metadataDir = resolve(asrPath, '..');
  const resultsDir = resolve(metadataDir, '..');
  const label = labelArg || basename(resultsDir).replace(/^results-/, '').replace(/^ggml-/, '');

  if (!existsSync(GROUND_TRUTH)) {
    console.error(`Ground truth not found: ${GROUND_TRUTH}`);
    process.exit(1);
  }

  console.log(`[llm-fix] Reading: ${asrPath}`);
  const raw = JSON.parse(readFileSync(asrPath, 'utf-8'));
  const { segments } = normalizeTranscription(raw);
  const promptInput = segmentsToPrompt(segments);

  console.log(`[llm-fix] ${segments.length} segs, sending ${promptInput.length} chars to ${model}...`);
  const t0 = performance.now();
  const fixed = await fixWithLLM(promptInput, { model, apiBase, domainHint });
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

  const fixedTexts = parseLines(fixed, segments.length);
  let resultSegments: any[];
  let resultText: string;
  let fallback = false;

  if (fixedTexts) {
    resultSegments = segments.map((s: any, i: number) => ({ ...s, text: fixedTexts[i] }));
    resultText = fixedTexts.join(' ');
    console.log(`  Done in ${elapsed}s, parsed ${fixedTexts.length} segs OK`);
  } else {
    resultSegments = segments;
    resultText = segments.map((s: any) => s.text).join(' ');
    fallback = true;
    console.log(`  Done in ${elapsed}s, PARSE FAILED, using original`);
  }

  const fixedFile = join(metadataDir, `wer-${label}-llm-fixed.json`);
  writeFileSync(fixedFile, JSON.stringify({
    audio_info: { duration: segments.length ? segments[segments.length - 1].end : 0 },
    result: { text: resultText, segments: resultSegments },
    _llm_fixed: true, _source: asrPath, _label: label,
  }, null, 2));

  const origFile = join(metadataDir, `wer-${label}.json`);
  const origResult = computeWER(GROUND_TRUTH, origFile);
  const fixedResult = computeWER(GROUND_TRUTH, fixedFile);

  const cerImpr = ((origResult.cer - fixedResult.cer) / Math.max(origResult.cer, 0.0001) * 100).toFixed(1);
  const werImpr = ((origResult.wer - fixedResult.wer) / Math.max(origResult.wer, 0.0001) * 100).toFixed(1);

  console.log(`\n=== LLM ASR 纠错: ${label} ===`);
  console.log(`             CER       WER`);
  console.log(`  原始:      ${(origResult.cer * 100).toFixed(2)}%    ${(origResult.wer * 100).toFixed(2)}%`);
  console.log(`  LLM 修正:  ${(fixedResult.cer * 100).toFixed(2)}%    ${(fixedResult.wer * 100).toFixed(2)}%`);
  console.log(`  改善:      ${cerImpr}%          ${werImpr}%`);
  console.log(`  Fallback:  ${fallback}`);
  console.log(`\nSaved: ${fixedFile}`);

  const summaryFile = join(metadataDir, `wer-${label}-llm-summary.json`);
  writeFileSync(summaryFile, JSON.stringify({
    label, fallback,
    cerBefore: origResult.cer, cerAfter: fixedResult.cer,
    werBefore: origResult.wer, werAfter: fixedResult.wer,
    cerImprovement: +(cerImpr),
    werImprovement: +(werImpr),
    segmentsBefore: segments.length,
    segmentsAfter: resultSegments.length,
    _source: asrPath,
  }, null, 2));
  console.log(`Saved: ${summaryFile}`);
}

main().catch(err => {
  console.error('Failed:', err);
  process.exit(1);
});
