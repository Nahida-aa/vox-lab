import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { Demucs } from '../../cli/src/ml/demucs/demucs';
import {
  RESULTS_DIR,
  REF_DIR,
  AUDIO_KEYS,
  audioDuration,
  round,
  type BenchmarkResult,
} from './bench-shared';

export async function benchmarkONNX(): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];
  const device = 'cpu';

  console.log('[ONNX] Loading model...');
  const t0 = performance.now();
  const model = new Demucs(undefined, { executionProvider: 'cpu', stems: ['drums', 'bass', 'other', 'vocals'] });
  await model.load();
  const loadTimeS = (performance.now() - t0) / 1000;
  console.log(`[ONNX] Model loaded in ${loadTimeS.toFixed(1)}s`);

  for (const key of AUDIO_KEYS) {
    const audioPath = join(REF_DIR, `${key}.wav`);
    if (!existsSync(audioPath)) {
      console.warn(`[ONNX] ${audioPath} not found, skipping`);
      continue;
    }
    const durationS = audioDuration(audioPath);
    console.log(`[ONNX] Processing ${key} (${durationS.toFixed(1)}s)...`);

    const t1 = performance.now();
    await model.separate(audioPath);
    const processTimeS = (performance.now() - t1) / 1000;

    results.push({
      engine: 'ort',
      device,
      audioKey: key,
      audioDurationS: round(durationS, 3),
      loadTimeS: round(loadTimeS, 3),
      processTimeS: round(processTimeS, 3),
      totalTimeS: round(loadTimeS + processTimeS, 3),
      rtf: round(processTimeS / durationS, 3),
    });

    console.log(`  RTF: ${results[results.length - 1].rtf}`);
  }

  return results;
}
