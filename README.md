# vox-lab

语音处理多后端实验与 Rust 化工作区：围绕 **ASR（whisper）**、**音源分离（demucs）**、
**TTS / VC（VoxCPM、CosyVoice）** 三条链路，对比 Rust（Burn / GGML / ONNX Runtime /
自研 WGSL 内核）、PyTorch 与云端运行时在不同设备（CPU / Vulkan / CUDA / ROCm …）上的推理表现，
并沉淀基准测试、问题排查记录与可发布的预构建资产。

架构与实现决策集中记录在 **[`docs/decisions/`](./docs/decisions/README.md)**
（ADR 一决策一文件，唯一真相）；本文件只描述现状，决策理由与被废弃的方案见该目录。

## 术语

| 词 | 含义 | 说明 |
|----|------|------|
| **Runtime（运行时）** | 一套推理实现 | 如 `burn`、`pytorch`、`cloud`。用户选的是「用哪套实现来跑」。 |
| **Device（设备）** | 计算后端 / 硬件目标 | `cpu` / `vulkan` / `cuda` / `rocm` / `metal`。即 Burn 里 backend 的概念，我们对外叫设备。 |
| **RuntimeConfig（运行配置）** | 运行时的参数集合 | 立项设想，**未落地**：无此类型，device 由编译期 feature 决定。见 [ADR 0002](./docs/decisions/0002-no-scheduler-compile-time-device.md)。 |

> 注：Burn 内部把 Vulkan/CUDA 称作 `Backend`；本项目对外统一称 **Device**，
> 内部映射到 Burn 的 backend 即可，不冲突。

## 设计原则（实际落地形态）

1. **不做中央调度层**：各实现可独立调用，互不编排。Rust 侧是
   「公共逻辑库 + 每个后端一个独立 CLI 二进制」——`demucs-burn` 产出
   `-tch` / `-wgpu`（另有未发布的 `-cuda` / `-rocm`）；`voxcpm-burn` 曾按
   `cpu` / `vulkan` / `wgpu` / `tch` 切分，现移出主 workspace 暂不构建
   （见 [ADR 0008](./docs/decisions/0008-demucs-tch-standalone-workspace.md)）。
   **它们不起 HTTP 服务**。HTTP 服务只在 Python 侧
   （`voxcpm_pytorch`，FastAPI + Gradio）。`packages/demucs_pytorch/` 是
   PyTorch 参考实现，仅用于校验 `demucs-tch`，不是服务。
   音源分离另有两条不依赖 Burn 的路径：`demucs-tch`（tch-rs / libtorch）与
   `demucs-wgpu`（自研 WGSL kernel + Vulkan，封装 `submodule/demucs-wgpu`），
   见 [ADR 0009](./docs/decisions/0009-demucs-tch-backend-builds.md)、
   [0010](./docs/decisions/0010-demucs-wgpu-backend.md)。
2. **Device 由构建产物决定，不暴露运行时开关**：Burn 侧每个后端由 Cargo feature
   门控（`cpu` / `wgpu` / `vulkan` / `tch` / `cuda` / `rocm`），一个二进制绑定一个
   Burn backend；tch 侧没有 feature，能力由构建时链接的 libtorch 决定（CPU 版恒为
   CPU，CUDA/ROCm 版自动用 GPU）；wgpu 侧按 adapter 自动选。CLI 都没有 `--device`
   参数。不存在「同一 Runtime 运行时切 device」的正交关系，也没有承载该选择的
   `RuntimeConfig` 类型。
3. **按后端分别落地、分别发版**：各后端独立基准测试（`packages/benchmark/`）
   与独立预构建资产（`.github/workflows/release-*.yml`），按实测结论选默认后端
   （音源分离：CPU 走 `demucs-tch`，GPU 走 `demucs-wgpu`；Burn 侧历史基线见
   `packages/demucs-burn/docs/README.md`）。
4. **对齐云端的 Gradio 契约**：本地 `voxcpm_pytorch` 复刻
   `voxcpm.modelbest.cn` 的 `/generate` 签名，客户端按同一 Gradio 协议对接，
   本地与云端靠「同签名」互换，而非另立一套标准。
5. **只做推理/部署**：假设已有模型权重，不含训练与微调。

> 决策背景与被废弃的立项设想（Runtime 自包含起服务、Device 运行时正交、
> `RuntimeConfig` 配置即调度）见 [ADR 0002](./docs/decisions/0002-no-scheduler-compile-time-device.md)、
> [0003](./docs/decisions/0003-rust-cli-python-http.md)。
> 原措辞保留在 ROADMAP 的 M0–M5（已标 ⛔ 历史规划）中。

## 架构总览（VoxCPM TTS 链路）

> 仓库还包含 ASR / 音源分离链路，见「仓库现状」。Rust 侧的 `voxcpm-burn` /
> `demucs-burn` 目前是**库**，不起 HTTP 服务；下图的「服务器」指 Python 侧。

```
                 ┌───────────────────────────────────────────────┐
   客户端 ──────►│  Gradio 协议：predict('/generate', [...10 参数]) │
   TS / Rust    │  （同一套契约，本地与云端同签名）                │
                 └────────────────┬──────────────────┬───────────┘
                                  │                  │
              ┌───────────────────┴────┐  ┌──────────┴─────────────────┐
              │ 本地 voxcpm_pytorch    │  │ 官方云端 voxcpm.modelbest.cn│
              │ FastAPI + Gradio        │  │ Gradio 6.10 应用            │
              │ 默认 127.0.0.1:19112    │  │                             │
              │ + /status /load-model   │  │                             │
              │   /unload-model + mDNS  │  │                             │
              └───────────┬────────────┘  └─────────────────────────────┘
                          │
                 submodule/VoxCPM (官方 PyTorch)
                 本地 cpu / cuda 加载权重推理

   客户端实现：
   - packages/voxcpm-cloud   TS: @gradio/client
                             Rust: gradio crate v0.4.2 (git tag, 支持 Gradio 4/5/6)
   - packages/vox-core       公共 wav 读写 / 重采样
```

关键点：**只有一套契约**（Gradio `/generate`），本地服务复刻云端签名，
因此客户端 `--target local|cloud` 只是换 endpoint，参数与返回解析完全一致。
没有中央调度层，各实现互不编排。

> 决策与废弃方案（曾设想本地用 OpenAI `/v1/audio/speech`、两套契约并存 +
> 适配层切换，从未实现）见 [ADR 0004](./docs/decisions/0004-gradio-generate-contract.md)。

### HTTP 契约：Gradio `/generate`（实测）

> ⚠️ 实测结论：官方 `https://voxcpm.modelbest.cn` **不是** OpenAI/vLLM-Omni 服务，
> 而是一个 **Gradio 6.10 应用**。本地 `voxcpm_pytorch` 复刻其 `/generate` 签名，
> 所以两者反而是同构的（同为 Gradio），并非「天然不同构」。
> 该结论如何推翻立项架构，见 [ADR 0004](./docs/decisions/0004-gradio-generate-contract.md)；
> Rust 客户端选型见 [ADR 0005](./docs/decisions/0005-gradio-rs-client.md)。

- **对接方式**：TS 用官方 `@gradio/client` SDK（`Client.connect(url)` +
  `client.predict('/generate', [...])`），参考音频用 `handle_file(localPath)` 由 SDK 自动上传；
  Rust 侧用 `gradio` crate（`git = JacobLinCool/gradio-rs, tag = v0.4.2`，
  `packages/voxcpm-cloud/Cargo.toml`），支持 Gradio 4/5/6，已修复 6.10
  `parameters[].label` 为 i18n map 的解析问题。**不推荐手搓 `/gradio_api/predict` HTTP**。
- `/generate` 位置参数顺序（以 `cloud.ts` 实测为准，优先级高于 `/gradio_api/info` 的字段名）：
  ```
  [text, controlInstruction, refFile, isUltimate, promptText, cfg, normalize, denoise, dit_steps, user_id]
  ```
  对应：文本 / Voice Design 指令 / 参考音频(FileData) / 终极克隆开关 / 精准克隆转写 / cfg(1–3,默认2) /
  归一化 / 降噪 / CFM步数(1–50,默认10) / 用户ID
- 本地服务 `generate()` 签名同上（`packages/voxcpm_pytorch/server.py`），
  返回 `(48000, np.ndarray)`；云端返回 Gradio `FileData`，
  `result.data[0].url` 即音频，`fetch` 后解析 WAV（默认 48kHz）。

## 子模块

| 子模块 | 作用 | 链路 |
|--------|------|------|
| `submodule/voxcpm-rs` | 纯 Rust / Burn 的 voxcpm2 推理（cpu/vulkan/wgpu） | TTS / Burn |
| `submodule/VoxCPM` | 官方 PyTorch 实现（app.py / cli / core），`torch>=2.5` | TTS / PyTorch 参考 |
| `submodule/demucs-rs` | Rust 侧 Demucs 分离实现 | 分离 |
| `submodule/demucs` | 官方 Demucs（PyTorch，`adefossez/demucs`） | 分离 / 权重与参考 |
| `submodule/demucs.cpp` | GGML / C++ Demucs 推理 | 分离 / GGML 基准 |
| `submodule/demucs-wgpu` | 纯 Rust + wgpu 的 HTDemucs（自研 WGSL kernel，CPU/Vulkan 双后端） | 分离 / GPU（fork: Nahida-aa） |
| `submodule/whisper.cpp` | whisper.cpp（钉在 LocalDub 同 commit，Vulkan 后端） | ASR |
| `submodule/CosyVoice` | 官方 CosyVoice（CosyVoice2/3） | TTS / VC 基准 |

> `voxcpm-rs` 已提供干净公共 API：`VoxCPM::from_local(dir,&device)` + `generate(text,opts)`，
> 按 feature 切 Burn 后端。

## 仓库现状

Cargo workspace 成员见根 `Cargo.toml`；`packages/` 下按职责分组：

| 分组 | 目录 | 说明 |
|------|------|------|
| Rust 运行时 | `voxcpm-burn`、`demucs-burn`、`voxcpm-cloud`、`vox-core`、`voice-cloning`、`util` | Burn / 云端对接 / 公共 wav·工具库 |
| Rust 分离后端 | `demucs-tch`（tch-rs CPU，可构建 cuda/rocm 变体）、`demucs-burn-tch`（Burn 0.21 对比基线）、`demucs-wgpu`（自研 WGSL / Vulkan） | 均不依赖主 workspace 的 Burn 版本，各自独立 workspace |
| Python 服务/工具 | `voxcpm_pytorch`、`demucs_pytorch`、`servers_py` | uv workspace；`voxcpm_pytorch` 分 `core` / `cli` / `server`，含精度与分段耗时实验 |
| 基准 | `benchmark/`（asr / ocr / separate / VC 等子目录与 `results/FINDINGS.md`） | 各后端 RTF / CER / 内存实测 |
| 研究记录 | `research/`（ROCm hang、模型加载耗时、ONNX NaN、ASR 回归等） | 问题排查与结论沉淀 |
| 经验文档 | `packages/demucs-burn/docs/` | cpu fallback、conv patch、权重来源、后端基准 |

音源分离各后端实测（同一段 33.878 s 音轨，`htdemucs`，RTF = 处理时间 / 音频时长，越小越快）：

| 路径 | 设备 | RTF | 备注 |
|------|------|-----|------|
| `demucs-wgpu` | Vulkan (780M) | **0.11** | 自研 WGSL kernel；`htdemucs_ft` 为 0.45 |
| `demucs-tch` | CPU (libtorch) | 1.03 | `htdemucs_ft` 为 4.74 |
| `demucs-wgpu` | CPU (host 路径) | 1.36 | 同一份代码的对比基线 |
| PyTorch + ROCm | 780M | 0.64 | 需 `HSA_OVERRIDE_GFX_VERSION`，两次运行结果不一致，不作生产 |

数值一致性与取舍见 [ADR 0010](./docs/decisions/0010-demucs-wgpu-backend.md)。

CI 发布见 `.github/workflows/`：`release-{whisper,demucs}-{linux,windows}.yml` 产出
`whisper-vulkan`、`demucs-burn-{tch,wgpu}`；`release-demucs-tch-{linux,windows}.yml`
按 backend 矩阵产出 `demucs-tch-{cpu,cuda,rocm}`（Windows 无 rocm；CPU 包平铺
libtorch，GPU 包只含 exe，libtorch 由使用者自备）。`demucs-wgpu` 目前仅在本地
构建（`packages/demucs-wgpu/justfile`），尚未接入发布流水线。

## 参考实现与实测结论

以下能力源自 LocalDub `packages/voxlab` 与相关 torch server，其中大部分已迁入本仓库
（见「仓库现状」）。契约实测结论仍然有效，保留备查：

### 1) `voxlab` 的 TS 多后端引擎（`engines/voxcpm/`）
`/home/aa/repos/env_ls/LocalDub/packages/voxlab/src/engines/voxcpm/` 已是一个**可运行的
voxcpm 多后端 TS 引擎**：

- `index.ts`：`createVoxCPM(backend, config)` 工厂 + `VoxCPMBackend` 枚举
  （`PYTORCH` / `CLOUD` / `ORT`）
- `types.ts`：统一接口 `TTSBackend { load(); generate(opts); dispose(); }`、
  `TTSGenerateOptions { text, referenceWavPath, cfgValue?, promptText?, maxPatches? }`、
  `TTSGenerateResult { samples: Float32Array, loadTimeSec, genTimeSec }`
- `cloud.ts`：**云端对接实测可用方案** —— 用 `@gradio/client` 的
  `Client.connect(url)` + `client.predict('/generate', [...positionalArgs])`，
  参考音频用 `handle_file(localPath)`（SDK 自动上传，无需手搓上传接口）。
  `/generate` 位置参数顺序（以 cloud.ts 实测为准）：
  `[text, controlInstruction, refFile, isUltimate, promptText, cfg, normalize, denoise, dit_steps, user_id]`
- `pth.ts`：本地 PyTorch（python 子进程）运行时参考
- `onnx-node.ts`：ONNX Runtime (cpu/webgpu) 运行时参考

### 2) `voxcpm_pytorch`（云端版的本地实现，Python/Gradio）
已迁入 `packages/voxcpm_pytorch/`：一个**本地 Gradio 服务，复刻了
`voxcpm.modelbest.cn` 的 `/generate` 签名**（含 `ultimate` / `prompt_text` /
`cfg_value` / `dit_steps`），用官方 `VoxCPM.from_pretrained` 在本地 cpu/cuda 跑，
并额外提供 `/status`、`/load-model`、`/unload-model` 与 mDNS 发现（默认端口 19112）。
**这意味着本地可直接起一个与云端同契约的服务**，用于确定性测试。
同批迁入的还有 `servers_py` 公共依赖；原 `demucs_torch_server` 已整理为
`packages/demucs_pytorch/`（PyTorch 参考验证工具）。

### 3) 本项目 `packages/voxcpm-cloud`（已实测跑通）
`packages/voxcpm-cloud/` 已作为独立测试模块落地，基于 `@gradio/client`：
- `connect()` + `predict('/generate', [...positionalArgs])`，参数顺序与 cloud.ts 一致
- `--target cloud` 打真云端、`--target local` 打本地 `voxcpm_pytorch`
- **实测结论**：真云端基础 TTS 与声音克隆（用 `assets/media/任务完成.wav`）均成功，
  产出 48kHz WAV（`tmp/voxcpm-cloud.wav` / `tmp/voxcpm-cloud-clone.wav`）

**结论**：落地时沿用了这些已验证的抽象，没有重复造 `TTSBackend` /
`TTSGenerateOptions`：`voxcpm-cloud` 用 `@gradio/client` 按 Gradio `/generate`
对接云端与本地 `voxcpm_pytorch`（同签名互换），`voxcpm-burn` /
`demucs-burn` 则是 Rust 库，不起 HTTP 服务。早期设想的 `runtime-burn` /
`runtime-pytorch` 命名与 OpenAI 契约均未采用，见「仓库布局（现状）」。

## 仓库布局（现状）

```
vox-lab/
├── Cargo.toml                    # Rust workspace（成员见文件内 members）
├── pyproject.toml                # uv workspace（torch servers + servers_py）
├── requirements-pytorch-cu128.txt
├── submodule/                    # 8 个上游子模块，见「子模块」表
├── packages/
│   ├── vox-core/                 # 公共 wav 读写 / 重采样等工具
│   ├── voxcpm-burn/              # VoxCPM Burn 运行时
│   ├── demucs-burn/              # Demucs Burn 运行时（tch / wgpu）
│   ├── demucs-tch/               # Demucs tch-rs 运行时（独立 workspace，cpu/cuda/rocm）
│   ├── demucs-burn-tch/          # Burn 0.21 + burn-tch 性能对比基线
│   ├── demucs-wgpu/              # 自研 WGSL kernel 的 Vulkan 推理封装
│   ├── voxcpm-cloud/             # 云端 Gradio 契约客户端（TS + Rust）
│   ├── voice-cloning/            # 声音克隆相关
│   ├── util/                     # cargo_build_cmd / cargo_build_bin 等构建辅助
│   ├── voxcpm_pytorch/           # VoxCPM 实验（core/cli）+ 本地 Gradio 服务
│   ├── demucs_pytorch/           # PyTorch 参考实现（校验 demucs-tch）
│   ├── servers_py/               # Python 服务公共依赖
│   ├── benchmark/                # ASR / 分离 / OCR / VC 基准与 results/
│   └── research/                 # 问题排查与实验记录
├── assets/  data/  e2e/  target/  tmp/
├── settings.example.json / .env.example   # 配置与秘密模板（提交）
└── .github/workflows/            # whisper / demucs 的 Linux + Windows 发布
```

> 早期规划的 `runtime-burn` / `runtime-pytorch` / `shared-proto` / `cli` /
> `ts-client` / `config` 等命名未采用，实际能力分散在上表各包中，
> 见 [ADR 0006](./docs/decisions/0006-submodules-and-runtimes.md)。
> 「无调度层」「只有 Python 侧起 HTTP 服务」见
> [0002](./docs/decisions/0002-no-scheduler-compile-time-device.md)、
> [0003](./docs/decisions/0003-rust-cli-python-http.md)。

## 配置系统

> ⚠️ **约定，未实现加载器**：`settings.example.json` / `.env.example` 两个模板存在，
> 但 `packages/config` 加载器与三层合并未落地（无 `RuntimeConfig` 类型）。
> 实际配置入口为各 CLI 的 clap 参数、Python 服务参数与环境变量
> （如 `DEMUCS_MODEL_DIR`）。决策见 [ADR 0007](./docs/decisions/0007-config-conventions.md)；
> 以下保留立项时的配置约定备查。

三层配置，优先级从低到高：**代码默认 < `settings.json` < 环境变量/秘密**。

### 1) 默认配置（代码内置，零配置可跑）
```jsonc
{
  "model_dir": "models/voxcpm2",
  "default_runtime": "burn",          // burn | pytorch | cloud
  "default_device": "cpu",            // cpu | vulkan | cuda | rocm
  "server": { "host": "127.0.0.1", "port": 0 },  // port=0 自动选空闲端口
  "log_level": "info"
}
```

### 2) `settings.json`（用户配置，非敏感）
可提交（或用 `settings.example.json` 作模板）。覆盖默认值，按运行时分段：
```jsonc
{
  "model_dir": "models/voxcpm2",
  "default_runtime": "burn",
  "runtimes": {
    "burn":    { "device": "vulkan" },
    "pytorch": { "device": "cuda" },
    "cloud":   { "api_url": "https://voxcpm.modelbest.cn" }
  },
  "server": { "host": "127.0.0.1", "port": 19112 }
}
```

### 3) 秘密配置（敏感，最小化，用 `.env`）
秘密配置**不做全量**，只用 `.env` 承载敏感/需覆盖的键值对（非敏感个性化项仍在 `settings.json`）：
- `VOXCPM_API_KEY` —— 云端鉴权（若 `api.modelbest.cn/v1` 需要；对应 `VoxCPMCloudConfig.apiKey`）
- `VOXCPM_CLOUD_API_URL` —— 覆盖云端地址（可选）
- `VOXCPM_MODEL_DIR` —— （可选）若想用环境变量覆盖权重目录

`.env` 可不存在（纯本地推理无需秘密），加载器跳过即可；不引入全量 `.secrets.json`。

**加载顺序**：`Default::default()` → merge `settings.json` → merge `.env`/环境变量（最高优先）。
运行时只读取 `config.runtimes.<自身>` 那段，互不影响（契合去中心化）。

### gitignore 约定
```
models/               # 大权重
settings.json         # 个人配置（差异大，忽略）
.env                  # 个人秘密（忽略）
```
入库模板：`settings.example.json`、`.env.example`（均不含真实值）。

## 里程碑

> 立项时的阶段划分，保留作规划记录；实际进展见「当前状态」与 ROADMAP。
> 其中 M0/M2 的 OpenAI `/v1/audio/speech` 契约**并未实现**，
> 见 [ADR 0004](./docs/decisions/0004-gradio-generate-contract.md)。

| 阶段 | 目标 |
|------|------|
| M0 | 仓库骨架 + 术语 + shared-proto（HTTP 契约 = OpenAI /v1/audio/speech） |
| M1 | burn 运行时：包装 voxcpm-rs，独立调用（cpu / vulkan） |
| M2 | burn 运行时：独立 HTTP 服务器（自包含，契约对齐云端） |
| M3 | TS 客户端：连本地 burn 服务器 / 云端，自动切换 |
| M4 | MVP 软件：CLI 一键起 burn 服务器 + 推理 |
| M5 | pytorch 运行时：独立调用 + 独立服务器（cpu / cuda） |
| M6 | 扩展 Device：burn 的 cuda/rocm、pytorch 的 rocm |
| M7 | 多模型扩展机制（注册表），为「更多模型」铺路 |

## 当前状态

- [x] 8 个子模块引入（VoxCPM / voxcpm-rs / Demucs 系 / demucs-wgpu / whisper.cpp / CosyVoice）
- [x] Rust workspace + 多个 Rust 运行时 crate（`voxcpm-burn` / `demucs-burn` / `voxcpm-cloud` / `vox-core` 等）
- [x] uv workspace + 本地 PyTorch / Gradio 服务（`*_torch_server`、`servers_py`）
- [x] 云端与本地统一的 Gradio `/generate` 契约实测、文档化（`voxcpm-cloud` + `voxcpm_pytorch`）
- [x] 配置约定落地：`.gitignore` / `settings.example.json` / `.env.example`
- [x] 基准测试与研究记录沉淀（`benchmark/` + `research/`）
- [x] Linux / Windows 预构建资产发布（whisper-vulkan、demucs-burn tch/wgpu、demucs-tch cpu）
- [x] 音源分离 GPU 路径落地：自研 WGSL kernel（Vulkan），htdemucs 3.6 s / 34 s 音频
- [x] LocalDub 侧主要能力的迁入（详见 ROADMAP「移植」章节）
- [ ] LocalDub 侧 `steps/tts/mod.rs` 依赖切换与 `packages/voxlab` 退役（跨仓库收尾）

> M0–M7 里程碑为立项时的规划视角，未逐项按原命名推进；跨仓库迁移与收尾进度
> 以 [ROADMAP.md](./ROADMAP.md) 与 task list 为准。架构决策见
> [`docs/decisions/`](./docs/decisions/README.md)。
