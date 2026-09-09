import { benchmarkGGML } from './ggml-cpu';
import { printSummary, saveResults } from './bench-shared';

async function main() {
  console.log('=== Demucs ggml Benchmark ===\n');
  const results = await benchmarkGGML();
  printSummary(results);
  saveResults(results, );
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
