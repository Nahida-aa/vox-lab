/**
 * voxcpm-cloud — 对接 voxcpm.modelbest.cn（Gradio）的独立测试模块。
 *
 * 契约（实测自 https://voxcpm.modelbest.cn，Gradio 6.10）：
 *   - 用 @gradio/client 的 Client.connect(url) + client.predict('/generate', [...])
 *   - /generate 位置参数顺序（以 LocalDub/voxlab cloud.ts 实测为准）：
 *     [text, controlInstruction, refFile, isUltimate, promptText,
 *      cfg, normalize, denoise, dit_steps, user_id]
 *   - 参考音频用 handle_file(localPath)，由 SDK 自动上传
 *   - 返回 result.data[0].url 即音频，fetch 后为 WAV（默认 48kHz）
 *
 * 该模块与本地 voxcpm_torch_server（复刻同签名）可互换使用：
 *   target=cloud -> https://voxcpm.modelbest.cn
 *   target=local -> http://127.0.0.1:19112
 */
import { Client, handle_file, type Client as GradioClient } from '@gradio/client';

export interface VoxCPMCloudOptions {
  text: string;
  /** Voice Design 音色描述（无参考音频时使用） */
  controlInstruction?: string;
  /** 参考音频本地路径（声音克隆） */
  referenceWavPath?: string;
  /** 终极克隆开关 */
  ultimate?: boolean;
  /** 参考音频的转写文本（精准克隆） */
  promptText?: string;
  /** 引导尺度 1.0–3.0 */
  cfgValue?: number;
  normalize?: boolean;
  refDenoise?: boolean;
  /** CFM 步数 1–50 */
  ditSteps?: number;
  userId?: string;
}

export interface VoxCPMCloudResult {
  samples: Float32Array;
  sampleRate: number;
  genTimeSec: number;
  audioUrl?: string;
}

export interface VoxCPMCloudConfig {
  apiUrl?: string;
}

const DEFAULT_CLOUD_URL = 'https://voxcpm.modelbest.cn';
const DEFAULT_LOCAL_URL = 'http://127.0.0.1:19112';

export class VoxCPMCloud {
  readonly name = 'cloud';
  private client?: GradioClient;
  private url: string;

  constructor(config: VoxCPMCloudConfig = {}) {
    this.url = config.apiUrl ?? DEFAULT_CLOUD_URL;
  }

  async connect(): Promise<void> {
    console.error(`[VoxCPMCloud] connecting ${this.url} ...`);
    this.client = await Client.connect(this.url);
    console.error(`[VoxCPMCloud] connected.`);
  }

  async dispose(): Promise<void> {
    this.client = undefined;
  }

  async generate(opts: VoxCPMCloudOptions): Promise<VoxCPMCloudResult> {
    if (!this.client) throw new Error('call connect() first');
    const t0 = performance.now();

    const refFile = opts.referenceWavPath ? handle_file(opts.referenceWavPath) : null;

    const args = [
      opts.text,
      opts.controlInstruction ?? '',
      refFile,
      opts.ultimate ?? false,
      opts.promptText ?? '',
      opts.cfgValue ?? 2.0,
      opts.normalize ?? false,
      opts.refDenoise ?? false,
      opts.ditSteps ?? 10,
      opts.userId ?? '',
    ];

    const result = await this.client.predict('/generate', args);
    const genTimeSec = (performance.now() - t0) / 1000;

    const data = result.data as Array<{ url?: string; path?: string }>;
    const audioFile = data[0];
    const audioUrl = audioFile?.url ?? audioFile?.path;
    if (!audioUrl) throw new Error('no audio url in response');

    const resp = await fetch(audioUrl);
    const buf = await resp.arrayBuffer();
    const { samples, sampleRate } = parseWav(buf);

    return { samples, sampleRate, genTimeSec, audioUrl };
  }
}

/** 解析 WAV（RIFF）为 Float32 PCM + 采样率。 */
export function parseWav(buf: ArrayBuffer): { samples: Float32Array; sampleRate: number } {
  const view = new DataView(buf);
  let offset = 12;
  let sampleRate = 48000;
  const readStr = (pos: number, len: number) =>
    Array.from(new Uint8Array(buf, pos, len), (c) => String.fromCharCode(c)).join('');

  while (offset + 8 <= buf.byteLength) {
    const chunkId = readStr(offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === 'fmt ') {
      sampleRate = view.getUint32(offset + 12, true);
    } else if (chunkId === 'data') {
      const dataStart = offset + 8;
      const pcmCount = Math.min(chunkSize, buf.byteLength - dataStart) >> 1;
      const int16 = new Int16Array(buf, dataStart, pcmCount);
      const samples = new Float32Array(pcmCount);
      for (let i = 0; i < pcmCount; i++) samples[i] = int16[i] / 32768;
      return { samples, sampleRate };
    }
    offset += 8 + chunkSize + (chunkSize & 1);
  }

  // 无 data chunk：整段当作裸 PCM
  const pcmCount = buf.byteLength >> 1;
  const int16 = new Int16Array(buf);
  const samples = new Float32Array(pcmCount);
  for (let i = 0; i < pcmCount; i++) samples[i] = int16[i] / 32768;
  return { samples, sampleRate: 48000 };
}

export { DEFAULT_CLOUD_URL, DEFAULT_LOCAL_URL };
