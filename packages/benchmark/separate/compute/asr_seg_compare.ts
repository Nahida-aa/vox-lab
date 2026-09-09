import { readFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const REPO_ROOT = resolve(__dirname, '..', '..', '..');
const RESULTS_DIR = join(__dirname, 'results');
const GROUND_TRUTH = resolve(REPO_ROOT, 'packages', 'benchmark', 'ref', 'metadata', 'srt_manual.json');

interface Segment { text: string; start: number; end: number }

interface VersionDef {
  label: string;
  origFile: string;
  fixedFile: string;
  origSegFile: string;
}

const VERSIONS: VersionDef[] = [
  { label: 'raw', origFile: 'wer-raw-video.json', fixedFile: 'wer-raw-llm-fixed.json', origSegFile: 'wer-raw-video.json' },
  { label: 'ggml', origFile: 'wer-ggml-shifts1-16bit.json', fixedFile: 'wer-ggml-llm-fixed.json', origSegFile: 'wer-ggml-shifts1-16bit.json' },
  { label: 'ort', origFile: 'wer-ort-video.json', fixedFile: 'wer-ort-llm-fixed.json', origSegFile: 'wer-ort-video.json' },
  { label: 'ps1', origFile: 'wer-pytorch-shifts1.json', fixedFile: 'wer-pytorch-s1-llm-fixed.json', origSegFile: 'wer-pytorch-shifts1.json' },
  { label: 'ps3', origFile: 'wer-pytorch-shifts3.json', fixedFile: 'wer-pytorch-s3-llm-fixed.json', origSegFile: 'wer-pytorch-shifts3.json' },
];

function loadJson(p: string): any {
  return JSON.parse(readFileSync(p, 'utf-8'));
}

function findGt(seg: Segment, gt: Segment[], tolerance = 4000): { match: Segment; idx: number; dist: number } | null {
  let best: { match: Segment; idx: number; dist: number } | null = null;
  for (let i = 0; i < gt.length; i++) {
    const dist = Math.abs(seg.start - gt[i].start);
    if (dist <= tolerance && (!best || dist < best.dist)) best = { match: gt[i], idx: i, dist };
  }
  return best;
}

function findAsr(gtSeg: Segment, asr: Segment[], tolerance = 4000): { match: Segment; idx: number; dist: number } | null {
  let best: { match: Segment; idx: number; dist: number } | null = null;
  for (let i = 0; i < asr.length; i++) {
    const dist = Math.abs(gtSeg.start - asr[i].start);
    if (dist <= tolerance && (!best || dist < best.dist)) best = { match: asr[i], idx: i, dist };
  }
  return best;
}

interface MatchInfo {
  exact: boolean;
  fixedCorrect?: boolean;
  fixedWrong?: boolean;
  origText?: string;
  fixedText?: string;
  gtText: string;
  start: number;
  dist: number;
}

function analyze(origSegs: Segment[], fixedSegs: Segment[], gtSegs: Segment[], label: string) {
  // For each GT segment, find the best ASR match (before and after LLM)
  const gtMatched = new Set<number>();
  const asrMatchedOrig = new Set<number>();
  const asrMatchedFixed = new Set<number>();

  const matches: { gtIdx: number; dist: number; origText: string; fixedText: string; gtText: string; fixedCorrect: boolean; fixedWrong: boolean }[] = [];

  // GT → ASR alignment
  for (let gi = 0; gi < gtSegs.length; gi++) {
    const g = gtSegs[gi];
    const origMatch = findAsr(g, origSegs);
    const fixedMatch = findAsr(g, fixedSegs);
    // Pick the best of orig/fixed
    const best = origMatch && fixedMatch
      ? (origMatch.dist <= fixedMatch.dist ? origMatch : fixedMatch)
      : (origMatch || fixedMatch);
    if (!best) continue;

    const gtText = g.text;
    const origText = origSegs[best.idx]?.text || '';
    const fixedText = fixedSegs[best.idx]?.text || '';
    const exactOrig = origText === gtText;
    const exactFixed = fixedText === gtText;
    // LLM fixed correct: orig was wrong, fixed is correct
    const fixedCorrect = !exactOrig && exactFixed;
    // LLM fixed wrong: orig was correct, fixed is wrong
    const fixedWrong = exactOrig && !exactFixed && fixedText !== origText;
    // LLM introduced new error: diff from orig and still wrong
    const sameWrong = !exactOrig && !exactFixed && origText !== fixedText;

    gtMatched.add(gi);
    asrMatchedOrig.add(best.idx);

    matches.push({ gtIdx: gi, dist: best.dist, origText, fixedText, gtText, fixedCorrect, fixedWrong });
  }

  // Count unmatched GT segments
  const gtUnmatched = gtSegs.length - gtMatched.size;
  // Count extra ASR segments (ASR segments not matched to any GT)
  const extraOrig = origSegs.length - asrMatchedOrig.size;
  const extraFixed = fixedSegs.length - asrMatchedFixed.size;

  const exactOrig = matches.filter(m => m.origText === m.gtText).length;
  const exactFixed = matches.filter(m => m.fixedText === m.gtText).length;
  const fixedCorrectCount = matches.filter(m => m.fixedCorrect).length;
  const fixedWrongCount = matches.filter(m => m.fixedWrong).length;
  const sameWrongCount = matches.filter(m => !m.fixedCorrect && !m.fixedWrong && m.origText !== m.fixedText && m.fixedText !== m.gtText).length;

  const avgDist = matches.length ? matches.reduce((a, m) => a + m.dist, 0) / matches.length : 0;

  return {
    gtSegs: gtSegs.length,
    origSegs: origSegs.length,
    fixedSegs: fixedSegs.length,
    matched: matches.length,
    gtUnmatched,
    extraOrig,
    exactOrig,
    exactFixed,
    fixedCorrectCount,
    fixedWrongCount,
    sameWrongCount,
    avgDist: +avgDist.toFixed(3),
    matches,
  };
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

function cer(a: string, b: string): number {
  if (!a && !b) return 0;
  return levenshtein(a, b) / Math.max(a.length, b.length);
}

function main() {
  const gt: { result: { text: string; segments: Segment[] } } = loadJson(GROUND_TRUTH);
  const gtSegs = gt.result.segments;

  console.log('=== ASR 分段人工标注对比 ===\n');
  console.log(`GT: ${gtSegs.length} 段, 全文 ${gt.result.text.split(/\s+/).length} 词\n`);

  const summary: any[] = [];

  for (const v of VERSIONS) {
    const origPath = join(RESULTS_DIR, v.origFile);
    const fixedPath = join(RESULTS_DIR, v.fixedFile);
    if (!existsSync(origPath) || !existsSync(fixedPath)) {
      console.warn(`  SKIP ${v.label}: ${v.origFile} or ${v.fixedFile} not found`);
      continue;
    }

    const origData = loadJson(origPath);
    const fixedData = loadJson(fixedPath);
    const origSegs: Segment[] = origData.result?.segments || [];
    const fixedSegs: Segment[] = fixedData.result?.segments || [];

    const origText = origData.result?.text || '';
    const fixedText = fixedData.result?.text || '';
    const gtFullText = gt.result.text || '';

    const r = analyze(origSegs, fixedSegs, gtSegs, v.label);

    // WER/CER contexts
    const cerOrig = cer(origText.replace(/\s+/g, ''), gtFullText.replace(/\s+/g, ''));
    const cerFixed = cer(fixedText.replace(/\s+/g, ''), gtFullText.replace(/\s+/g, ''));

    summary.push({ label: v.label, ...r, cerOrig: +(cerOrig * 100).toFixed(2), cerFixed: +(cerFixed * 100).toFixed(2) });
  }

  console.log('=== 全版本对比 ===');
  console.log('版本\tGT段\t原始段\t修正段\t匹配\tGT未匹配\t多余段\t精确(原)\t精确(修)\tLLM修正\tLLM加错\t同错\t时间偏差\tCER原\tCER修\t改善');
  for (const s of summary) {
    const cerImpr = s.cerOrig > 0 ? ((s.cerOrig - s.cerFixed) / s.cerOrig * 100).toFixed(1) : '0.0';
    console.log(
      `${s.label.padEnd(6)}\t${s.gtSegs}\t${s.origSegs}\t${s.fixedSegs}\t${s.matched}\t${s.gtUnmatched}\t${s.extraOrig}\t${s.exactOrig}\t${s.exactFixed}\t${s.fixedCorrectCount}\t${s.fixedWrongCount}\t${s.sameWrongCount}\t${(s.avgDist / 1000).toFixed(3)}s\t${s.cerOrig}%\t${s.cerFixed}%\t${cerImpr}%`,
    );
  }

  console.log('\n=== LLM 修正 + 加错 + 同错 明细 (每版本 top 10) ===');
  for (const s of summary) {
    console.log(`\n--- ${s.label} ---`);
    const fixedCorrect = s.matches.filter((m: any) => m.fixedCorrect);
    const fixedWrong = s.matches.filter((m: any) => m.fixedWrong);
    const sameWrong = s.matches.filter((m: any) => {
      const isFixedCorrect = m.fixedCorrect;
      const isFixedWrong = m.fixedWrong;
      return !isFixedCorrect && !isFixedWrong && m.origText !== m.fixedText && m.fixedText !== m.gtText;
    });

    if (fixedCorrect.length) {
      console.log(`  ✅ LLM 修正正确 (${fixedCorrect.length}):`);
      for (const m of fixedCorrect.slice(0, 5)) {
        console.log(`    [${m.gtIdx}] '${m.origText}' → '${m.fixedText}' (GT: '${m.gtText}')`);
      }
      if (fixedCorrect.length > 5) console.log(`    ... and ${fixedCorrect.length - 5} more`);
    }

    if (fixedWrong.length) {
      console.log(`  ❌ LLM 改错 (${fixedWrong.length}):`);
      for (const m of fixedWrong.slice(0, 5)) {
        console.log(`    [${m.gtIdx}] '${m.origText}' → '${m.fixedText}' (GT: '${m.gtText}')`);
      }
      if (fixedWrong.length > 5) console.log(`    ... and ${fixedWrong.length - 5} more`);
    }

    if (sameWrong.length) {
      console.log(`  ⚠️  LLM 改后仍错 (${sameWrong.length}):`);
      for (const m of sameWrong.slice(0, 5)) {
        console.log(`    [${m.gtIdx}] '${m.origText}' → '${m.fixedText}' (GT: '${m.gtText}')`);
      }
      if (sameWrong.length > 5) console.log(`    ... and ${sameWrong.length - 5} more`);
    }
  }

  console.log('\n=== GT 未匹配段（各版本 ASR 完全漏掉的） ===');
  for (const s of summary) {
    if (s.gtUnmatched === 0) continue;
    // Find unmatched GT segments
    const matchedSet = new Set(s.matches.map((m: any) => m.gtIdx));
    for (let gi = 0; gi < s.gtSegs; gi++) {
      if (!matchedSet.has(gi)) {
        console.log(`  ${s.label}: GT[${gi}] '${gtSegs[gi].text}' @${(gtSegs[gi].start / 1000).toFixed(2)}s`);
      }
    }
  }
}

main();
