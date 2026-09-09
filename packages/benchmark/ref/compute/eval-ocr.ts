import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, join, dirname } from 'node:path';

const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..');
const DEFAULT_GT = resolve(REPO_ROOT, 'packages', 'benchmark', 'ref', 'metadata', 'ocr_manual.json');

interface Segment {
  text: string;
  start: number;
  end: number;
}

export function loadSegments(path: string, scaleMs = 1): Segment[] {
  const d = JSON.parse(readFileSync(path, 'utf-8'));
  return d.result.segments.map((s: any) => ({
    text: s.text.trim(),
    start: s.start * scaleMs,
    end: s.end * scaleMs,
  }));
}

export function normalizeForCER(s: string): string {
  let t = s;
  t = t.replace(/师父/g, '师傅');
  t = t.replace(/\s+/g, '');
  t = t.replace(/[。，！？、；：""''「」【】《》（）\.,!?;:'"()\[\]{}\u2018\u2019\u201c\u201d\u3000\u3001\u3002\uff01\uff0c\uff1f\uff1a\uff1b\u2026—～~\-]/g, '');
  t = t.replace(/([\d零一二两三四五六七八九十百千万亿])+/g, '#');
  return t;
}

function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] : Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1;
  return dp[m][n];
}

export function computeCER(ref: string, hyp: string): number {
  if (!ref && !hyp) return 0;
  return levenshtein(ref, hyp) / Math.max(ref.length, 1);
}

function iou(a: Segment, b: Segment): number {
  const interStart = Math.max(a.start, b.start);
  const interEnd = Math.min(a.end, b.end);
  const intersection = Math.max(0, interEnd - interStart);
  const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
  return union > 0 ? intersection / union : 0;
}

function computeGaps(segs: Segment[]) {
  const gaps: number[] = [];
  let totalCovered = 0;
  for (let i = 0; i < segs.length; i++) {
    totalCovered += segs[i].end - segs[i].start;
    if (i > 0) gaps.push(segs[i].start - segs[i - 1].end);
  }
  const posGaps = gaps.filter(g => g > 0);
  const audioDuration = segs.length > 0 ? segs[segs.length - 1].end - segs[0].start : 0;
  return {
    coverMs: totalCovered,
    coverRatio: audioDuration > 0 ? totalCovered / audioDuration : 0,
    gapCount: posGaps.length,
    gapTotalMs: posGaps.reduce((a, b) => a + b, 0),
    gapAvgMs: posGaps.length > 0 ? posGaps.reduce((a, b) => a + b, 0) / posGaps.length : 0,
    gapMaxMs: posGaps.length > 0 ? Math.max(...posGaps) : 0,
  };
}

export function analyzeOffsets(gt: Segment[], hyp: Segment[]) {
  const offsetsStart: number[] = [];
  const offsetsEnd: number[] = [];
  const matchedHyp = new Set<number>();
  const missed: Segment[] = [];
  const details: { gtIdx: number; hypIdx: number; iou: number; soff: number; eoff: number; gtText: string; hypText: string }[] = [];

  for (let gi = 0; gi < gt.length; gi++) {
    const g = gt[gi];
    let bestIou = 0;
    let bestIdx = -1;
    for (let i = 0; i < hyp.length; i++) {
      const o = iou(g, hyp[i]);
      if (o > bestIou) {
        bestIou = o;
        bestIdx = i;
      }
    }
    if (bestIou > 0) {
      matchedHyp.add(bestIdx);
      const h = hyp[bestIdx];
      const soff = h.start - g.start;
      const eoff = h.end - g.end;
      offsetsStart.push(soff);
      offsetsEnd.push(eoff);
      details.push({ gtIdx: gi, hypIdx: bestIdx, iou: bestIou, soff, eoff, gtText: g.text, hypText: h.text });
    } else {
      missed.push(g);
    }
  }

  let fpCount = 0;
  const fps: { hypIdx: number; text: string; start: number; end: number }[] = [];
  for (let i = 0; i < hyp.length; i++) {
    if (!matchedHyp.has(i)) {
      let hasOverlap = false;
      for (const g of gt) {
        if (iou(hyp[i], g) > 0) { hasOverlap = true; break; }
      }
      if (!hasOverlap) {
        fpCount++;
        fps.push({ hypIdx: i, text: hyp[i].text, start: hyp[i].start, end: hyp[i].end });
      }
    }
  }

  const sortedStart = [...offsetsStart].sort((a, b) => a - b);
  const sortedEnd = [...offsetsEnd].sort((a, b) => a - b);
  const mid = Math.floor(sortedStart.length / 2);
  const median = (arr: number[]) => arr.length % 2 === 1 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;

  const gapInfo = computeGaps(hyp);

  return {
    nGt: gt.length,
    nHyp: hyp.length,
    matched: details.length,
    ...gapInfo,
    detectionRate: details.length > 0 ? details.length / gt.length : 0,
    startOffsetMean: offsetsStart.length > 0 ? offsetsStart.reduce((a, b) => a + b, 0) / offsetsStart.length : 0,
    startOffsetMedian: sortedStart.length > 0 ? median(sortedStart) : 0,
    startOffsetMae: offsetsStart.length > 0 ? offsetsStart.reduce((a, b) => a + Math.abs(b), 0) / offsetsStart.length : 0,
    endOffsetMean: offsetsEnd.length > 0 ? offsetsEnd.reduce((a, b) => a + b, 0) / offsetsEnd.length : 0,
    endOffsetMedian: sortedEnd.length > 0 ? median(sortedEnd) : 0,
    endOffsetMae: offsetsEnd.length > 0 ? offsetsEnd.reduce((a, b) => a + Math.abs(b), 0) / offsetsEnd.length : 0,
    falsePositives: fpCount,
    falsePositiveDetails: fps,
    missedSegments: missed,
    details,
  };
}

interface Options {
  gtPath: string;
  hypPath: string;
  label: string;
  outPath?: string;
}

export function main(opts: Options & { hypScaleMs?: number }) {
  const gt = loadSegments(opts.gtPath, 1);
  const hyp = loadSegments(opts.hypPath, opts.hypScaleMs ?? 1000);

  const gtFull = gt.map(s => s.text).join('');
  const hypFull = hyp.map(s => s.text).join('');

  const gtNorm = normalizeForCER(gtFull);
  const hypNorm = normalizeForCER(hypFull);

  const rawCer = computeCER(gtFull.replace(/\s+/g, ''), hypFull.replace(/\s+/g, ''));
  const normCer = computeCER(gtNorm, hypNorm);

  const offsetResult = analyzeOffsets(gt, hyp);

  const result = {
    label: opts.label,
    gtPath: opts.gtPath,
    hypPath: opts.hypPath,
    raw: {
      cer: +(rawCer * 100).toFixed(2),
      refChars: gtFull.replace(/\s+/g, '').length,
      hypChars: hypFull.replace(/\s+/g, '').length,
    },
    normalized: {
      cer: +(normCer * 100).toFixed(2),
      refChars: gtNorm.length,
      hypChars: hypNorm.length,
    },
    segments: {
      nGt: offsetResult.nGt,
      nHyp: offsetResult.nHyp,
      matched: offsetResult.matched,
      detectionRate: +(offsetResult.detectionRate * 100).toFixed(1),
      startOffsetMeanMs: +offsetResult.startOffsetMean.toFixed(1),
      startOffsetMedianMs: +offsetResult.startOffsetMedian.toFixed(1),
      startOffsetMaeMs: +offsetResult.startOffsetMae.toFixed(1),
      endOffsetMeanMs: +offsetResult.endOffsetMean.toFixed(1),
      endOffsetMedianMs: +offsetResult.endOffsetMedian.toFixed(1),
      endOffsetMaeMs: +offsetResult.endOffsetMae.toFixed(1),
      falsePositives: offsetResult.falsePositives,
      missedCount: offsetResult.missedSegments.length,
      coverRatio: +(offsetResult.coverRatio * 100).toFixed(1),
      gapCount: offsetResult.gapCount,
      gapTotalMs: offsetResult.gapTotalMs,
      gapAvgMs: +offsetResult.gapAvgMs.toFixed(0),
      gapMaxMs: offsetResult.gapMaxMs,
    },
  };

  const out: string[] = [];
  out.push('='.repeat(72));
  out.push(`  ASR Evaluation: ${opts.label}`);
  out.push('='.repeat(72));
  out.push('');
  out.push(`  GT:     ${opts.gtPath}`);
  out.push(`  Hyp:    ${opts.hypPath}`);
  out.push('');
  out.push('  ── Text ──');
  out.push(`  Raw CER:         ${result.raw.cer}%  (ref=${result.raw.refChars} hyp=${result.raw.hypChars})`);
  out.push(`  Normalized CER:  ${result.normalized.cer}%  (ref=${result.normalized.refChars} hyp=${result.normalized.hypChars})`);
  out.push('');

  const impr = result.raw.cer - result.normalized.cer;
  out.push(`  CER improvement from normalization: ${impr > 0 ? '+' : ''}${impr.toFixed(2)}ppt`);
  out.push('');
  out.push('  ── Segments ──');
  out.push(`  GT segments:  ${result.segments.nGt}`);
  out.push(`  Hyp segments: ${result.segments.nHyp}`);
  out.push(`  Matched:      ${result.segments.matched}  (${result.segments.detectionRate}%)`);
  out.push(`  Missed:       ${result.segments.missedCount}`);
  out.push(`  False pos:    ${result.segments.falsePositives}`);
  out.push(`  Start offset: mean=${result.segments.startOffsetMeanMs >= 0 ? '+' : ''}${result.segments.startOffsetMeanMs}ms  median=${result.segments.startOffsetMedianMs >= 0 ? '+' : ''}${result.segments.startOffsetMedianMs}ms  mae=${result.segments.startOffsetMaeMs}ms`);
  out.push(`  End offset:   mean=${result.segments.endOffsetMeanMs >= 0 ? '+' : ''}${result.segments.endOffsetMeanMs}ms  median=${result.segments.endOffsetMedianMs >= 0 ? '+' : ''}${result.segments.endOffsetMedianMs}ms  mae=${result.segments.endOffsetMaeMs}ms`);
  out.push(`  Cover:  ${result.segments.coverRatio}%  (${result.segments.gapCount} gaps, total ${(result.segments.gapTotalMs/1000).toFixed(1)}s, avg ${result.segments.gapAvgMs}ms, max ${result.segments.gapMaxMs}ms)`);

  if (offsetResult.missedSegments.length > 0) {
    out.push('');
    out.push('  ── Missed GT Segments ──');
    for (const m of offsetResult.missedSegments) {
      out.push(`    [${(m.start).toFixed(0)}-${(m.end).toFixed(0)}] "${m.text}"`);
    }
  }

  if (offsetResult.falsePositives > 0) {
    out.push('');
    out.push('  ── False Positive Hyp Segments ──');
    for (const fp of offsetResult.falsePositiveDetails) {
      out.push(`    [${fp.start.toFixed(0)}-${fp.end.toFixed(0)}] "${fp.text}"`);
    }
  }

  out.push('');
  out.push('  ── Caveats ──');
  out.push('  • Timestamp offset ≠ timing error. ASR tends to skip filler words');
  out.push('    at segment boundaries (啊/嗯/哎/哈/哦/呀) and low-volume speech.');
  out.push('    A positive start offset or negative end offset commonly indicates');
  out.push('    the ASR simply omitted edge filler, not that timing is wrong.');
  out.push('  • Missed segments are nearly always filler-only (啊, 哈哈哈, 啊啊).');
  out.push('    These are merged into neighboring segments by whisper\'s decoder.');
  out.push('  • Detection rate is IOU-based; a 0-IOU match means zero overlap.');
  out.push('');
  out.push('  ── Normalization Rules ──');
  out.push('  1. 师父 → 师傅 (unify homophones)');
  out.push('  2. Strip whitespace & punctuation');
  out.push('  3. Numerals (Arabic + Chinese) → # placeholder');
  out.push('');
  out.push('='.repeat(72));
  out.push('');

  console.log(out.join('\n'));

  if (opts.outPath) {
    writeFileSync(opts.outPath, JSON.stringify(result, null, 2), 'utf-8');
    console.log(`  Written to: ${opts.outPath}\n`);
  }

  return result;
}

function runBatch(baseDir: string, gtPath: string, hypScaleMs = 1, hypFile = 'ocr.json') {
  const { readdirSync, statSync } = require('fs');
  const entries = readdirSync(baseDir, { withFileTypes: true });
  const results: any[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const hypPath = join(baseDir, entry.name, 'metadata', hypFile);
    try { statSync(hypPath); } catch { continue; }
    const label = entry.name;
    const r = main({ gtPath, hypPath, label, hypScaleMs });
    results.push(r);
  }

  console.log('');
  console.log('='.repeat(72));
  console.log('  BATCH SUMMARY');
  console.log('='.repeat(72));
  console.log('');
  console.log('  Label'.padEnd(22) + ' normCER  det%   segs  s_off_mn  e_off_mn  s_mae  e_mae  miss  fp  cover   gaps  g_avg');
  console.log('  ' + '-'.repeat(96));

  const sorted = [...results].sort((a, b) => a.normalized.cer - b.normalized.cer);
  for (const r of sorted) {
    const label = r.label.padEnd(20);
    const cer = r.normalized.cer.toFixed(2).padStart(7);
    const det = r.segments.detectionRate.toString().padStart(5);
    const segs = `${r.segments.nHyp}`.padStart(4);
    const som = (r.segments.startOffsetMeanMs >= 0 ? '+' : '') + r.segments.startOffsetMeanMs.toFixed(0);
    const eom = (r.segments.endOffsetMeanMs >= 0 ? '+' : '') + r.segments.endOffsetMeanMs.toFixed(0);
    const smae = `${r.segments.startOffsetMaeMs.toFixed(0)}`.padStart(5);
    const emae = `${r.segments.endOffsetMaeMs.toFixed(0)}`.padStart(5);
    const miss = `${r.segments.missedCount}`.padStart(4);
    const fp = `${r.segments.falsePositives}`.padStart(3);
    const cover = r.segments.coverRatio.toFixed(1).padStart(5);
    const gaps = `${r.segments.gapCount}`.padStart(4);
    const gAvg = `${r.segments.gapAvgMs}`.padStart(5);
    console.log(`  ${label} ${cer}%  ${det}%  ${segs}  ${som}ms  ${eom}ms  ${smae}ms  ${emae}ms  ${miss}  ${fp}  ${cover}%  ${gaps}  ${gAvg}ms`);
  }

  console.log('');
}

if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.length < 1 || args[0] === '--help') {
    console.error('Usage:');
    console.error('  bun eval-ocr.ts <hyp.json> [gt.json] [--label <label>]');
    console.error('  bun eval-ocr.ts --batch <results_dir> [gt.json]');
    console.error(`  Default GT: ${DEFAULT_GT}`);
    process.exit(1);
  }

  if (args[0] === '--batch') {
    const baseDir = resolve(args[1]);
    let gtPath = DEFAULT_GT;
    let hypFile = 'ocr.json';
    let batchScaleMs = 1;
    for (let i = 2; i < args.length; i++) {
      if (args[i] === '--hyp-file') hypFile = args[++i];
      else if (args[i] === '--ms') batchScaleMs = 1;
      else if (!args[i].startsWith('--')) gtPath = resolve(args[i]);
    }
    runBatch(baseDir, gtPath, batchScaleMs, hypFile);
    process.exit(0);
  }

  const hypPath = resolve(args[0]);
  let gtPath = DEFAULT_GT;
  let label: string | undefined;
  let hypInMs = false;
  let outPath: string | undefined;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === '--label') label = args[++i];
    else if (args[i] === '--ms') hypInMs = true;
    else if (args[i] === '--write') outPath = join(dirname(hypPath), 'ocr_summary.json');
    else if (!args[i].startsWith('--')) gtPath = resolve(args[i]);
  }

  if (!label) label = hypPath.split('/').slice(-3, -1).join('/');

  main({ gtPath, hypPath, label, hypScaleMs: hypInMs ? 1 : 1000, outPath });
  process.exit(0);
}
