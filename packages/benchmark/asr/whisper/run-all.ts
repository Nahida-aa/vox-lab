import { benchmarkFasterWhisper } from './run-faster-whisper';
import { benchmarkWhisperPytorch } from './run-whisper-pytorch';
import { benchmarkWhisperGGUF } from './run-whisper-gguf';
import { benchmarkWhisperCli } from './run-whisper-cli';
import { printSummary, mergeResults, type BenchmarkResult } from './bench-shared';

const RESULTS_FILE = 'whisper-bench.json';

async function main() {
  const all: BenchmarkResult[] = [];

  console.log('=== ASR Whisper Benchmark ===\n');

  console.log('--- faster-whisper ---');
  const fw = await benchmarkFasterWhisper();
  all.push(...fw);

  console.log('\n--- whisper-pytorch ---');
  const wp = await benchmarkWhisperPytorch();
  all.push(...wp);

  console.log('\n--- whisper.cpp / GGUF ---');
  const gguf = await benchmarkWhisperGGUF();
  all.push(...gguf);

  console.log('\n--- whisper-cli (GPU HIPBLAS) ---');
  const cli = await benchmarkWhisperCli();
  all.push(...cli);

  printSummary(all);
  mergeResults(all, RESULTS_FILE);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
