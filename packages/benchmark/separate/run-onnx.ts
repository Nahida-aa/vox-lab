import { benchmarkONNX } from './onnx-cpu';
import { printSummary, saveResults } from './bench-shared';

async function main() {
  console.log('=== Demucs ONNX CPU Benchmark ===\n');
  const results = await benchmarkONNX();
  printSummary(results);
  saveResults(results);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
