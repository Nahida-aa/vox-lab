import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..");
const RELEASE_DIR = join(REPO_ROOT, "target", "release");

type Backend = "wgpu";

interface BurnOptions {
  backend?: Backend;
  model?: string;
  tasksMax?: number;
  warmup?: boolean;
  timeout?: number;
}

interface BurnResult {
  loadTimeS: number;
  warmupTimeS: number;
  genTimeS: number;
}

export function runBurn(input: string, outDir: string, opts: BurnOptions = {}): BurnResult {
  const _backend: Backend = opts.backend ?? "wgpu";
  const binPath = join(RELEASE_DIR, "demucs-burn-wgpu");

  if (!existsSync(binPath)) {
    throw new Error(`Binary not found: ${binPath}.`);
  }

  const args: string[] = [];
  if (opts.warmup) args.push("--warmup");
  if (opts.model) args.push("--model", opts.model);
  if (opts.tasksMax) args.push("--tasks-max", String(opts.tasksMax));
  args.push(input, outDir);

  const result = spawnSync(binPath, args, {
    env: process.env,
    timeout: opts.timeout ?? 600_000,
    encoding: "utf-8",
  });

  if (result.status !== 0) {
    throw new Error(`demucs-burn-wgpu failed (exit ${result.status}):\n${result.stderr}`);
  }

  const stdout = result.stdout ?? "";
  const parse = (key: string) => {
    const m = stdout.match(new RegExp(`${key}:\\s*([\\d.]+)`));
    return m ? parseFloat(m[1]) : 0;
  };

  return {
    loadTimeS: parse("Benchmark-Load-Time"),
    warmupTimeS: parse("Benchmark-Warmup-Time"),
    genTimeS: parse("Benchmark-Gen-Time"),
  };
}

if (import.meta.path === process.argv[1]) {
  const args = process.argv.slice(2);
  const input = args.find((a) => !a.startsWith("--"));
  const outDir = args[args.indexOf(input!) + 1];
  if (!input || !outDir) {
    console.error("Usage: bun wrapper.ts <input.wav> <outDir> [--backend wgpu] [--warmup]");
    process.exit(1);
  }
  const backendIdx = args.indexOf("--backend");
  const backend: Backend = backendIdx >= 0 ? (args[backendIdx + 1] as Backend) : "wgpu";
  const warmup = args.includes("--warmup");

  try {
    const r = runBurn(input, outDir, { backend, warmup });
    console.log(`Load: ${r.loadTimeS.toFixed(3)}s`);
    if (r.warmupTimeS > 0) console.log(`Warmup: ${r.warmupTimeS.toFixed(3)}s`);
    console.log(`Generate: ${r.genTimeS.toFixed(3)}s`);
  } catch (e: any) {
    console.error(e.message);
    process.exit(1);
  }
}
