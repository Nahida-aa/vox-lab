/**
 * 独立测试脚本：验证 voxcpm-cloud 对接真实/本地 Gradio 服务。
 *
 * 用法：
 *   tsx src/test.ts --target cloud        # 测真云端 https://voxcpm.modelbest.cn
 *   tsx src/test.ts --target local        # 测本地 voxcpm_torch_server (127.0.0.1:19112)
 *   tsx src/test.ts --target local --ref  # 用参考音频做声音克隆
 *
 * 声音克隆参考音频：仓库内 assets/media/任务完成.wav（或 出现错误.wav）
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { VoxCPMCloud, DEFAULT_CLOUD_URL, DEFAULT_LOCAL_URL, parseWav } from "./index";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "../../..");
const REF_WAV = resolve(REPO_ROOT, "assets/media/任务完成.wav");

function parseArgs(argv: string[]) {
  const a = { target: "cloud", ref: false, text: "你好，这是一段语音克隆测试。" };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--target") a.target = argv[++i] ?? a.target;
    else if (argv[i] === "--ref") a.ref = true;
    else if (argv[i] === "--text") a.text = argv[++i] ?? a.text;
  }
  return a;
}

function writeWav(path: string, samples: Float32Array, sampleRate: number) {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * bytesPerSample, 28);
  buf.writeUInt16LE(bytesPerSample, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(dataSize, 40);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE((s * 32767) | 0, 44 + i * 2);
  }
  writeFileSync(path, buf);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const url = args.target === "local" ? DEFAULT_LOCAL_URL : DEFAULT_CLOUD_URL;
  console.error(`[test] target=${args.target} url=${url}`);
  console.error(`[test] text=${JSON.stringify(args.text)} ref=${args.ref ? REF_WAV : "(none)"}`);

  const cloud = new VoxCPMCloud({ apiUrl: url });
  await cloud.connect();

  const opts: Parameters<typeof cloud.generate>[0] = { text: args.text };
  if (args.ref) {
    opts.referenceWavPath = REF_WAV;
    opts.promptText = "任务完成。"; // 参考音频的转写（精准克隆）
  }

  const result = await cloud.generate(opts);
  console.error(
    `[test] genTimeSec=${result.genTimeSec.toFixed(2)} sampleRate=${result.sampleRate} samples=${result.samples.length}`,
  );

  const out = resolve(REPO_ROOT, "tmp", `voxcpm-${args.target}${args.ref ? "-clone" : ""}.wav`);
  writeWav(out, result.samples, result.sampleRate);
  console.error(`[test] wrote ${out}`);
  console.error(`[test] OK`);
}

main().catch((e) => {
  console.error("[test] FAILED:", e);
  process.exit(1);
});
