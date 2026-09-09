import { benchmarkWhisperPytorch } from './run-whisper-pytorch';
import { printSummary, saveResults } from './bench-shared';

const RESULTS_FILE = 'whisper-pytorch.json';

async function main() {
  console.log('=== ASR whisper-pytorch Benchmark ===\n');
  const results = await benchmarkWhisperPytorch();
  printSummary(results);
  saveResults(results, RESULTS_FILE);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
