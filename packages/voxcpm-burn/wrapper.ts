import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

const REPO_ROOT = resolve(import.meta.dirname, '..', '..');
const RELEASE_DIR = join(REPO_ROOT, 'target', 'release');

type Backend = 'vulkan' | 'wgpu' | 'cpu' | 'tch';

interface VoxCpmOptions {
  backend?: Backend;
  modelDir?: string;
  timesteps?: number;
  cfg?: number;
  maxLen?: number;
  warmup?: boolean;
  timeout?: number;
}

interface VoxCpmResult {
  loadTimeS: number;
  warmupTimeS: number;
  genTimeS: number;
}

function findLibtorchPath(): string | null {
  const buildDir = join(RELEASE_DIR, 'build');
  if (!existsSync(buildDir)) return null;
  for (const dir of readdirSync(buildDir)) {
    if (!dir.startsWith('torch-sys-')) continue;
    const libDir = join(buildDir, dir, 'out', 'libtorch', 'libtorch', 'lib');
    if (existsSync(join(libDir, 'libtorch_cpu.so'))) return libDir;
  }
  return null;
}

export function runVoxCpm(text: string, output: string, opts: VoxCpmOptions = {}): VoxCpmResult {
  const backend: Backend = opts.backend ?? 'vulkan';
  const binName = `voxcpm-burn-${backend}`;
  const binPath = join(RELEASE_DIR, binName);

  if (!existsSync(binPath)) {
    throw new Error(`Binary not found: ${binPath}.`);
  }

  const args: string[] = [];
  if (opts.warmup) args.push('--warmup');
  if (opts.modelDir) args.push('--model-dir', opts.modelDir);
  if (opts.timesteps) args.push('--timesteps', String(opts.timesteps));
  if (opts.cfg) args.push('--cfg', String(opts.cfg));
  if (opts.maxLen) args.push('--max-len', String(opts.maxLen));
  args.push(text, output);

  const env: Record<string, string> = { ...process.env as Record<string, string> };
  env.CUBECL_AUTOTUNE_LEVEL = 'minimal';

  if (backend === 'tch') {
    const libtorchLib = findLibtorchPath();
    if (!libtorchLib) {
      throw new Error('libtorch not found. Build tch binary first.');
    }
    env.LD_LIBRARY_PATH = [libtorchLib, env.LD_LIBRARY_PATH].filter(Boolean).join(':');
  }

  const result = spawnSync(binPath, args, { env, timeout: opts.timeout ?? 600_000, encoding: 'utf-8' });

  if (result.status !== 0) {
    throw new Error(`voxcpm-burn-${backend} failed (exit ${result.status}):\n${result.stderr}`);
  }

  const stdout = result.stdout ?? '';
  const parse = (key: string) => {
    const m = stdout.match(new RegExp(`${key}:\\s*([\\d.]+)`));
    return m ? parseFloat(m[1]) : 0;
  };

  return {
    loadTimeS: parse('Benchmark-Load-Time'),
    warmupTimeS: parse('Benchmark-Warmup-Time'),
    genTimeS: parse('Benchmark-Gen-Time'),
  };
}

if (import.meta.path === process.argv[1]) {
  const args = process.argv.slice(2);
  const text = args.find(a => !a.startsWith('--'));
  const output = args[args.indexOf(text!) + 1];
  if (!text || !output) {
    console.error('Usage: bun wrapper.ts <text> <output.wav> [--backend vulkan|wgpu|cpu|tch] [--timesteps N] [--cfg N] [--max-len N] [--warmup]');
    process.exit(1);
  }
  const backendIdx = args.indexOf('--backend');
  const backend: Backend = backendIdx >= 0 ? args[backendIdx + 1] as Backend : 'vulkan';
  const timestepsIdx = args.indexOf('--timesteps');
  const timesteps = timestepsIdx >= 0 ? parseInt(args[timestepsIdx + 1]) : undefined;
  const cfgIdx = args.indexOf('--cfg');
  const cfg = cfgIdx >= 0 ? parseFloat(args[cfgIdx + 1]) : undefined;
  const maxLenIdx = args.indexOf('--max-len');
  const maxLen = maxLenIdx >= 0 ? parseInt(args[maxLenIdx + 1]) : undefined;
  const warmup = args.includes('--warmup');

  try {
    const r = runVoxCpm(text, output, { backend, timesteps, cfg, maxLen, warmup });
    console.log(`Load: ${r.loadTimeS.toFixed(3)}s`);
    if (r.warmupTimeS > 0) console.log(`Warmup: ${r.warmupTimeS.toFixed(3)}s`);
    console.log(`Generate: ${r.genTimeS.toFixed(3)}s`);
  } catch (e: any) {
    console.error(e.message);
    process.exit(1);
  }
}
