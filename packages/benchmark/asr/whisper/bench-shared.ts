import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

export const RESULTS_DIR = join(__dirname, 'results');
export const VIDEO_PATH = resolve(__dirname, '..', '..', '..', '..', 'packages', 'benchmark', 'ref', 'media', 'video_source.mp4');
export const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..');

const isWin = process.platform === 'win32';
export const PYTHON_BIN = join(REPO_ROOT, '.venv', isWin ? 'Scripts' : 'bin', isWin ? 'python.exe' : 'python');

export interface BenchmarkResult {
  engine: string;
  device: string;
  computeType?: string;
  audioDurationS: number;
  loadTimeS: number;
  processTimeS: number;
  totalTimeS: number;
  rtf: number;
  text?: string;
}

export function round(v: number, d: number): number {
  const f = 10 ** d;
  return Math.round(v * f) / f;
}

export function saveResults(results: BenchmarkResult[], filename: string) {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const path = join(RESULTS_DIR, filename);
  writeFileSync(path, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to ${path}`);
}

export function mergeResults(results: BenchmarkResult[], filename: string) {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const path = join(RESULTS_DIR, filename);
  let all: BenchmarkResult[] = [];
  try {
    all = JSON.parse(readFileSync(path, 'utf-8'));
  } catch {}
  all.push(...results);
  writeFileSync(path, JSON.stringify(all, null, 2));
  console.log(`\nResults merged into ${path}`);
}

export function printSummary(results: BenchmarkResult[]) {
  console.log('\n=== ASR Benchmark Summary ===');
  console.log('Engine\t\t\tDevice\t\tDur(s)\tLoad(s)\tProc(s)\tRTF');
  for (const r of results) {
    const engine = r.computeType
      ? `${r.engine} (${r.computeType})`
      : r.engine;
    console.log(
      `${engine.padEnd(20)}\t${r.device.padEnd(8)}\t${r.audioDurationS.toFixed(1)}\t${r.loadTimeS.toFixed(1)}\t${r.processTimeS.toFixed(1)}\t${r.rtf.toFixed(3)}`,
    );
  }
}
