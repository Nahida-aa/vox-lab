import { benchmarkWhisperGGUF } from './run-whisper-gguf';
import { printSummary, saveResults } from './bench-shared';

const RESULTS_FILE = 'whisper-gguf.json';

async function main() {
  console.log('=== whisper.cpp GGUF Benchmark ===\n');
  const results = await benchmarkWhisperGGUF();
  printSummary(results);
  saveResults(results, RESULTS_FILE);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
