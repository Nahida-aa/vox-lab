import { benchmarkFasterWhisper } from './run-faster-whisper';
import { printSummary, saveResults } from './bench-shared';

const RESULTS_FILE = 'faster-whisper.json';

async function main() {
  console.log('=== ASR faster-whisper Benchmark ===\n');
  const results = await benchmarkFasterWhisper();
  printSummary(results);
  saveResults(results, RESULTS_FILE);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
