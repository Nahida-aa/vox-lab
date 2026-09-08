# tts-vc (voxcpm2 多后端推理)

让使用者通过 **Rust 库 + 软件**，使用 **voxcpm2** 模型进行 TTS / VC 推理，
支持多种计算设备（CPU / Vulkan / CUDA / ROCm / Metal …）与多种运行时
（Burn、PyTorch、云端 `https://voxcpm.modelbest.cn`）。

> 阶段：**架构规划 + 起步**。当前仅有 `submodule/voxcpm-rs` 子模块作为 Burn 运行时的基础。

## 术语

| 词 | 含义 | 说明 |
|----|------|------|
| **Runtime（运行时）** | 一套推理实现 | 如 `burn`、`pytorch`、`cloud`。用户选的是「用哪套实现来跑」。 |
| **Device（设备）** | 计算后端 / 硬件目标 | `cpu` / `vulkan` / `cuda` / `rocm` / `metal`。即 Burn 里 backend 的概念，我们对外叫设备。 |
| **RuntimeConfig（运行配置）** | 运行时的参数集合 | 权重路径、device 选择、并发、采样等。**暂不做统一调度层**，配置即调度。 |

> 注：Burn 内部把 Vulkan/CUDA 称作 `Backend`；本项目对外统一称 **Device**，
> 内部映射到 Burn 的 backend 即可，不冲突。

## 设计原则（去中心化）

1. **每个 Runtime 自包含**：能独立被调用，也能独立启动一个 HTTP 服务器。
   不先做集中式「调度中心」。
2. **Runtime 与 Device 正交**：
   - `Runtime=burn` 可跑在 `cpu` / `vulkan`（及未来的 `cuda` / `rocm`）。
   - `Runtime=pytorch` 可跑在 `cpu` / `cuda` / `rocm`。
   - `Runtime=cloud` 设备由云端决定，本地只发请求。
3. **先做好各 Runtime 的各 Device**，再考虑统一调度。
4. **HTTP 接口各 Runtime 自对齐云端**：本地 burn/pytorch 服务器与
   `voxcpm.modelbest.cn` 暴露相同契约，TS 侧可无缝切换。
5. **只做推理/部署**：假设已有 voxcpm2 权重。

## 架构总览（去中心化）

```
                 ┌─────────────────────────────────────────────┐
   用户 / TS ───►│  统一 HTTP 契约 = OpenAI /v1/audio/speech      │
                 │  （vLLM-Omni 所用标准，云端同构）              │
                 └──────────────┬──────────────┬───────────────┘
                                │              │
              ┌─────────────────┴──┐   ┌────────┴───────────────┐
              │ burn 运行时服务器   │   │ pytorch 运行时服务器    │  本地各自独立启动
              │ Device: cpu/vulkan │   │ Device: cpu/cuda/rocm  │
              └─────────┬──────────┘   └──────────┬─────────────┘
                        │                          │
                  submodule/voxcpm-rs        submodule/VoxCPM
                  (Burn 实现, 已是基础)       (官方 PyTorch, 参考)
                        │
                  ┌─────┴───────────────────────┐
                  │ 云端端点 (Gradio 协议)        │  https://voxcpm.modelbest.cn
                  │ /gradio_api/predict /generate│
                  └──────────────────────────────┘
```

关键点：**没有中央调度层**。每个运行时自己加载模型、自己起服务。
`RuntimeConfig` 承载选择逻辑（选哪个运行时、哪个设备），但不强制统一编排。

### HTTP 契约：两套并存（已实测）

> ⚠️ 实测结论：官方 `https://voxcpm.modelbest.cn` **不是** OpenAI/vLLM-Omni 服务，
> 而是一个 **Gradio 6.10 应用**。因此「本地与云端天然同构」不成立，需分别对接。

**1) 本地运行时服务器（burn / pytorch）→ 采用 OpenAI TTS 标准 `POST /v1/audio/speech`**
- 这是我们**自己的**契约选择（vLLM-Omni 也用此标准，生态兼容 openai SDK）。
- 基础：`{"model","input","voice","response_format","speed"}`
- 扩展（声音克隆 / Voice Design）：`prompt_audio` / `prompt_text` / `cfg` / `timesteps` / `seed`
- 音频响应（wav/mp3）、错误格式对齐 OpenAI。

**2) 官方云端 `voxcpm.modelbest.cn` → Gradio 协议（实测所得）**
- **对接方式**：用官方 `@gradio/client` SDK（`Client.connect(url)` + `client.predict('/generate', [...])`），
  参考音频用 `handle_file(localPath)` 由 SDK 自动上传（见 `LocalDub/voxlab` 的 `cloud.ts`）。
  **不推荐手搓 `/gradio_api/predict` HTTP**。
- `/generate` 位置参数顺序（以 `cloud.ts` 实测为准，优先级高于 `/gradio_api/info` 的字段名）：
  ```
  [text, controlInstruction, refFile, isUltimate, promptText, cfg, normalize, denoise, dit_steps, user_id]
  ```
  对应：文本 / Voice Design 指令 / 参考音频(FileData) / 终极克隆开关 / 精准克隆转写 / cfg(1–3,默认2) /
  归一化 / 降噪 / CFM步数(1–50,默认10) / 用户ID
- 返回：Gradio `FileData`，`result.data[0].url` 即音频，`fetch` 后解析 WAV（默认 48kHz）。
- `runtime-cloud`（TS）专门按此协议对接，与本地 OpenAI 契约**不直接互通**。

**TS 客户端的「本地/云端切换」= 切换契约适配层**，不是仅改 endpoint。

## 子模块

| 子模块 | 作用 | 对应运行时 |
|--------|------|-----------|
| `submodule/voxcpm-rs` | 纯 Rust / Burn 的 voxcpm2 推理（cpu/vulkan/wgpu） | `runtime-burn` 的基础依赖/参考 |
| `submodule/VoxCPM` | 官方 PyTorch 实现（app.py / cli / core），`torch>=2.5` | `runtime-pytorch` 的参考实现 |

> `voxcpm-rs` 已提供干净公共 API：`VoxCPM::from_local(dir,&device)` + `generate(text,opts)`，
> 按 feature 切 Burn 后端。**先以它打通 burn 运行时**，pytorch 运行时后做。

## 已有参考实现（重要，优先复用）

`/home/aa/repos/env_ls/LocalDub/packages/` 下已有可运行参考，应作为本项目直接参考而非从零设计：

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

### 2) `voxcpm_torch_server`（云端版的本地实现，Python/Gradio）
`/home/aa/repos/env_ls/LocalDub/packages/voxcpm_torch_server/server.py` 是一个**本地
Gradio 服务，复刻了 `voxcpm.modelbest.cn` 的 `/generate` 签名**（含 `ultimate` /
`prompt_text` / `cfg_value` / `dit_steps`），用官方 `VoxCPM.from_pretrained` 在本地
cpu/cuda 跑，并额外提供 `/status`、`/load-model`、`/unload-model` 与 mDNS 发现
（默认端口 19112）。**这意味着本地可直接起一个与云端同契约的服务**，用于确定性测试。

### 3) 本项目 `packages/voxcpm-cloud`（已实测跑通）
`packages/voxcpm-cloud/` 已作为独立测试模块落地，基于 `@gradio/client`：
- `connect()` + `predict('/generate', [...positionalArgs])`，参数顺序与 cloud.ts 一致
- `--target cloud` 打真云端、`--target local` 打本地 `voxcpm_torch_server`
- **实测结论**：真云端基础 TTS 与声音克隆（用 `assets/media/任务完成.wav`）均成功，
  产出 48kHz WAV（`tmp/voxcpm-cloud.wav` / `tmp/voxcpm-cloud-clone.wav`）

**结论**：TS 侧规划与我们高度一致（去中心化多后端 + 统一接口）。本项目应：
1. Rust 侧做 `runtime-burn` / `runtime-pytorch`（本地服务器）；
2. TS 侧在 `LocalDub/voxlab` 的 `TTSBackend` 抽象基础上扩展，云端继续用 `@gradio/client`，
   本地走本项目 Rust 服务器的 OpenAI `/v1/audio/speech` 契约；
3. 不重复造 `TTSBackend` / `TTSGenerateOptions` 这类已验证的接口定义。

## 仓库布局（规划）

```
tts-vc/
├── submodule/
│   ├── voxcpm-rs/        # Burn 运行时基础（已引入，v0.5.0）
│   └── VoxCPM/           # 官方 PyTorch 实现（已引入，参考用）
├── models/               # 模型权重（gitignore，默认 model_dir）
│   └── voxcpm2/
│       ├── burn/         # Burn/voxcpm-rs 格式
│       └── pytorch/      # PyTorch/HF 格式
├── settings.example.json # 配置模板（提交，注明所有字段+默认值）
├── settings.json         # 个人真实配置（gitignore，从 example 复制）
├── .env.example          # 秘密变量模板（提交，敏感项留空）
├── .env                  # 个人真实秘密（gitignore，可不存在）
├── packages/
│   ├── config/           # 配置加载器：默认 < settings.json < 环境变量/secret
│   ├── runtime-burn/     # 包装 voxcpm-rs：独立调用 + 独立 HTTP 服务器
│   │   └─ 支持 Device: cpu / vulkan (后续 cuda/rocm)
│   ├── runtime-pytorch/  # 独立运行时：pyo3 + torch，Device: cpu/cuda/rocm
│   ├── runtime-cloud/    # 云端端点封装（薄客户端，与 HTTP 契约对齐）
│   ├── shared-proto/     # 各 Runtime 共享的 HTTP 请求/响应类型（对齐云端）
│   ├── cli/              # 软件：启动某运行时服务器 / 直接推理
│   └── ts-client/        # TS 客户端：按 endpoint 连本地/云端
```

## 配置系统

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

- [x] 引入 `submodule/voxcpm-rs`（Burn 实现基础，支持 cpu/vulkan/wgpu）
- [x] 引入 `submodule/VoxCPM`（官方 PyTorch 实现，参考用）
- [x] 配置约定落地：`.gitignore` / `settings.example.json` / `.env.example`
- [ ] M0 仓库骨架（Cargo workspace + packages/config + packages/shared-proto）
- [ ] 任意运行时端到端跑通
- [ ] 本地服务器
- [ ] TS 客户端
- [ ] MVP 软件

见 task list 跟踪进度。
