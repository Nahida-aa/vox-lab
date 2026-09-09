import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { VoxCPMBackend } from '@repo/voxlab';
import { runBenchmark, RESULTS_DIR } from './bench-shared';

async function main() {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const results = await runBenchmark({
    backend: VoxCPMBackend.PYTORCH,
    device: 'cpu',
    config: { python: '/home/aa/repos/learn_ls/YouDub-webui/.venv/bin/python' },
  });
  const outPath = join(RESULTS_DIR, 'pth-cpu.json');
  writeFileSync(outPath, JSON.stringify(results, null, 2), 'utf-8');
  console.log(`\nSaved to ${outPath}`);
}

main().catch(console.error);
