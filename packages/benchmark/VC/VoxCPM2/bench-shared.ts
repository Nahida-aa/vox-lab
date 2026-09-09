import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import {
  createVoxCPM,
  VoxCPMBackend,
  writeWav,
} from '@repo/voxlab';
import type { TTSBackend, TTSGenerateResult, VoxCPMNodeConfig, VoxCPMPythonConfig } from '@repo/voxlab';

export const RESULTS_DIR = join(__dirname, 'results');
export const REF_WAV = join(__dirname, 'ref.wav');

export interface BenchmarkResult {
  engine: string;
  device: string;
  text_key: string;
  text_len: number;
  load_time_s: number;
  generate_time_s: number;
  total_time_s: number;
  output_samples: number;
  output_duration_s: number;
  auto_patches: number;
  rtf: number;
}

export const TEXTS: Record<string, string> = {
  short: '今天天气真不错。',
  medium: '人工智能正在改变我们的生活和工作方式。从自然语言处理到计算机视觉，AI技术取得了显著进展。',
  long: '近年来，人工智能技术发展迅速，在自然语言处理、计算机视觉、语音识别等领域都取得了突破性进展。深度学习模型的不断完善，使得AI系统在理解和生成人类语言方面表现出色。同时，大语言模型的出现更是推动了整个行业的变革，为各行各业带来了新的机遇和挑战。未来，随着算力的提升和算法的优化，人工智能将在更多领域发挥重要作用。',
};

export function round(v: number, d: number): number {
  const f = 10 ** d;
  return Math.round(v * f) / f;
}

export interface BenchmarkConfig {
  backend: VoxCPMBackend;
  device?: string;
  config?: VoxCPMNodeConfig | VoxCPMPythonConfig;
}

export async function runBenchmark(cfg: BenchmarkConfig): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];
  const label = `${cfg.backend}${cfg.device ? '-' + cfg.device : ''}`;

  mkdirSync(RESULTS_DIR, { recursive: true });

  const t0 = performance.now();
  const model: TTSBackend = createVoxCPM(cfg.backend, cfg.config);
  await model.load();
  const loadTime = (performance.now() - t0) / 1000;

  for (const [textKey, text] of Object.entries(TEXTS)) {
    console.log(`\n[${label}] ${textKey}...`);
    const result: TTSGenerateResult = await model.generate({ text, referenceWavPath: REF_WAV, cfgValue: 2.0 });

    const loadTimeTotal = loadTime + result.loadTimeSec;
    const genTime = result.genTimeSec;

    // Save audio
    const wavPath = join(RESULTS_DIR, `${label}-${textKey}.wav`);
    writeWav(result.samples, wavPath, 48000);

    const outDur = result.samples.length / 48000;
    const r: BenchmarkResult = {
      engine: cfg.backend,
      device: cfg.device ?? '',
      text_key: textKey,
      text_len: text.length,
      load_time_s: round(loadTimeTotal, 3),
      generate_time_s: round(genTime, 3),
      total_time_s: round(loadTimeTotal + genTime, 3),
      output_samples: result.samples.length,
      output_duration_s: round(outDur, 3),
      auto_patches: Math.ceil(result.samples.length / 7680),
      rtf: round(genTime / outDur, 3),
    };
    console.log(JSON.stringify(r));
    results.push(r);
  }

  await model.dispose();
  return results;
}

/** Run all available backends and return combined results */
export async function runAllBackends(): Promise<Record<string, BenchmarkResult[]>> {
  const all: Record<string, BenchmarkResult[]> = {};
  const backends: BenchmarkConfig[] = [
    { backend: VoxCPMBackend.ORT, device: 'cpu', config: { executionProvider: 'cpu' } },
    { backend: VoxCPMBackend.PYTORCH, device: 'cpu', config: { python: '/home/aa/repos/learn_ls/YouDub-webui/.venv/bin/python' } },
  ];

  for (const cfg of backends) {
    try {
      all[cfg.backend] = await runBenchmark(cfg);
    } catch (err) {
      console.error(`[${cfg.backend}] FAILED:`, err);
    }
  }
  return all;
}
