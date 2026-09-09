import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { runAllBackends, RESULTS_DIR } from './bench-shared';

async function main() {
  mkdirSync(RESULTS_DIR, { recursive: true });
  console.log('=== VoxCPM Multi-Engine Benchmark ===\n');

  const all = await runAllBackends();

  const summaryPath = join(RESULTS_DIR, 'summary.json');
  writeFileSync(summaryPath, JSON.stringify(all, null, 2), 'utf-8');
  console.log(`\nSummary saved to ${summaryPath}`);
  console.log(`Audio outputs saved to ${RESULTS_DIR}/`);
}

main().catch(console.error);
