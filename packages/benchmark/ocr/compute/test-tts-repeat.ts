import { VoxCPMCloud, writeWav } from '@repo/voxlab';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..');
const VOCALS_DIR = resolve(REPO_ROOT, 'packages/benchmark/ocr/results/sep-sidechain-asr-vad-v6-th02-guided-ocr-so-ts0.45-end2fps/segments/vocals');
const TMP_DIR = resolve(REPO_ROOT, 'packages', 'tmp');

console.log({ REPO_ROOT, VOCALS_DIR, TMP_DIR, __dirname });

const text = '阁下剑法精妙绝伦';
const promptText = text;

const singleRef = resolve(VOCALS_DIR, '0016.wav');
const doubleRef = resolve(TMP_DIR, 'vocals_0016_x2.wav');
console.log({ singleRef, doubleRef });

const voxcpm = new VoxCPMCloud();
await voxcpm.load();

// Test 1: single reference (baseline)
console.log('\n=== Single reference ===');
const { samples: s1, genTimeSec: t1 } = await voxcpm.generate({
  text,
  referenceWavPath: singleRef,
  promptText,
});
writeWav(s1, resolve(TMP_DIR, 'tts_single_ref.wav'), 48000);
console.log(`Duration: ${(s1.length / 48000).toFixed(3)}s | Gen: ${t1.toFixed(1)}s`);

// Test 2: doubled reference
console.log('\n=== Doubled reference ===');
const { samples: s2, genTimeSec: t2 } = await voxcpm.generate({
  text,
  referenceWavPath: doubleRef,
  promptText,
});
writeWav(s2, resolve(TMP_DIR, 'tts_double_ref.wav'), 48000);
console.log(`Duration: ${(s2.length / 48000).toFixed(3)}s | Gen: ${t2.toFixed(1)}s`);

await voxcpm.dispose();
console.log('\nDone. Listen to:\n  packages/tmp/tts_single_ref.wav\n  packages/tmp/tts_double_ref.wav');
