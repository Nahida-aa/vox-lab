import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

export const RESULTS_DIR = join(__dirname, 'results');
export const REF_DIR = join(__dirname, 'ref');

export interface BenchmarkResult {
  engine: string;
  device: string;
  audioKey: string;
  audioDurationS: number;
  loadTimeS: number;
  processTimeS: number;
  totalTimeS: number;
  rtf: number;
}

export const AUDIO_KEYS = ['short', 'medium', 'long'] as const;

export function readWavHeader(path: string): {
  sampleRate: number;
  numChannels: number;
  bitsPerSample: number;
  dataSize: number;
} {
  const buf = readFileSync(path);
  if (buf.length < 12 || buf.toString('ascii', 0, 4) !== 'RIFF') {
    throw new Error('Not a valid WAV file');
  }
  let offset = 12;
  let sampleRate = 0;
  let numChannels = 0;
  let bitsPerSample = 0;
  let dataSize = 0;
  while (offset + 8 <= buf.length) {
    const chunkId = buf.toString('ascii', offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    if (chunkId === 'fmt ') {
      numChannels = buf.readUInt16LE(offset + 10);
      sampleRate = buf.readUInt32LE(offset + 12);
      bitsPerSample = buf.readUInt16LE(offset + 22);
    } else if (chunkId === 'data') {
      dataSize = chunkSize;
      break;
    }
    offset += 8 + chunkSize + (chunkSize % 2);
  }
  return { sampleRate, numChannels, bitsPerSample, dataSize };
}

export function audioDuration(path: string): number {
  const h = readWavHeader(path);
  if (!h.sampleRate || !h.dataSize) return 0;
  const bytesPerSample = h.bitsPerSample / 8;
  return h.dataSize / (h.sampleRate * h.numChannels * bytesPerSample);
}

export function round(v: number, d: number): number {
  const f = 10 ** d;
  return Math.round(v * f) / f;
}

export function saveResults(results: BenchmarkResult[]) {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const path = join(RESULTS_DIR, `separate-bench.json`);
  writeFileSync(path, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to ${path}`);
}

export function printSummary(results: BenchmarkResult[]) {
  console.log('\n=== Separate Benchmark Summary ===');
  console.log('Engine\t\tDevice\t\tAudio\t\tDur(s)\tLoad(s)\tProc(s)\tRTF');
  for (const r of results) {
    console.log(
      `${r.engine.padEnd(12)}\t${r.device.padEnd(8)}\t${r.audioKey.padEnd(8)}\t${r.audioDurationS.toFixed(1)}\t${r.loadTimeS.toFixed(1)}\t${r.processTimeS.toFixed(1)}\t${r.rtf.toFixed(3)}`,
    );
  }
}
