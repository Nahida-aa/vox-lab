import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { VoxCPMBackend } from '@repo/voxlab';
import { runBenchmark, RESULTS_DIR } from './bench-shared';

async function main() {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const results = await runBenchmark({
    backend: VoxCPMBackend.ORT,
    device: 'webgpu',
    config: { executionProvider: 'webgpu' },
  });
  const outPath = join(RESULTS_DIR, 'onnx-nodejs-webgpu.json');
  const { writeFileSync } = await import('node:fs');
  writeFileSync(outPath, JSON.stringify(results, null, 2), 'utf-8');
  console.log(`\nSaved to ${outPath}`);
}

main().catch(console.error);
