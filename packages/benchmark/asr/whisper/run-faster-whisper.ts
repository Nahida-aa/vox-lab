import { spawnSync } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import {
  RESULTS_DIR,
  VIDEO_PATH,
  REPO_ROOT,
  PYTHON_BIN,
  round,
  type BenchmarkResult,
} from './bench-shared';

const SCRIPT = join(REPO_ROOT, 'packages', 'cli', 'scripts', 'asr', 'run.py');

interface LoadConfig {
  label: string;
  device: string;
  args: string[];
}

const CONFIGS: LoadConfig[] = [
  { label: 'GPU (float16)', device: 'cuda', args: [] },
  { label: 'CPU (float32)', device: 'cpu',  args: ['--cpu', '--compute-type', 'float32'] },
  { label: 'CPU (int8)',    device: 'cpu',  args: ['--cpu'] },
];

function measureLoadTime(extraArgs: string[]): number {
  const t0 = performance.now();
  const result = spawnSync(PYTHON_BIN, [SCRIPT, '--benchmark-load', ...extraArgs], {
    timeout: 300_000,
  });
  if (result.status !== 0) {
    throw new Error(`--benchmark-load failed: ${result.stderr?.toString().slice(-300)}`);
  }
  return (performance.now() - t0) / 1000;
}

export async function benchmarkFasterWhisper(): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  for (const cfg of CONFIGS) {
    console.log(`[faster-whisper] ${cfg.label} — measuring load time...`);

    let loadTimeS: number;
    try {
      loadTimeS = round(measureLoadTime(cfg.args), 3);
      console.log(`  Load: ${loadTimeS}s`);
    } catch (err) {
      console.error(`  Load failed: ${err}`);
      continue;
    }

    console.log(`[faster-whisper] ${cfg.label} — transcribing...`);
    const outDir = join(RESULTS_DIR, `fw-${cfg.device}-${Date.now()}`);
    mkdirSync(outDir, { recursive: true });

    const t1 = performance.now();
    const procArgs = [SCRIPT, VIDEO_PATH, outDir, 'en', ...cfg.args];
    const result = spawnSync(PYTHON_BIN, procArgs, { timeout: 600_000 });

    if (result.status !== 0) {
      console.error(`  Transcribe failed: ${result.stderr?.toString().slice(-300)}`);
      rmSync(outDir, { recursive: true, force: true });
      continue;
    }

    const processTimeS = round((performance.now() - t1) / 1000, 3);
    const computeType = cfg.label.match(/\((\w+)\)/)?.[1] ?? 'int8';

    results.push({
      engine: 'faster-whisper',
      device: cfg.device,
      computeType,
      audioDurationS: 170,
      loadTimeS,
      processTimeS,
      totalTimeS: round(loadTimeS + processTimeS, 3),
      rtf: round(processTimeS / 170, 3),
    });

    console.log(`  RTF: ${results[results.length - 1].rtf}`);

    rmSync(outDir, { recursive: true, force: true });
  }

  return results;
}
