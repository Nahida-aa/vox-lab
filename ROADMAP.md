# 路线图 (ROADMAP)

> 方向：**去中心化**——每个 Runtime（burn / pytorch / cloud）自包含，
> 能独立调用、独立起服务器；暂不做统一调度层，用 `RuntimeConfig` 承载选择。
> 任务进度以 task list 为准。

## 术语
- **Runtime（运行时）**：一套推理实现（burn / pytorch / cloud）。
- **Device（设备）**：计算后端（cpu / vulkan / cuda / rocm / metal）。
  Burn 内部称 backend，本项目对外统一称 Device。
- **RuntimeConfig（运行配置）**：权重路径、device、并发、采样等参数。
  替代早期设想的「调度中心」。

## 阶段 M0 — 骨架 + HTTP 契约
**目标**：定好目录、术语、`shared-proto`（各 Runtime 共享的请求/响应类型）。

- Cargo workspace 初始化（packages/ 下各 crate）
- `shared-proto`：**本地运行时用 OpenAI TTS 标准 `POST /v1/audio/speech`** 作为契约
  （这是我们的选择，vLLM-Omni 也用此标准，生态兼容 openai SDK）
  - 基础字段：`model` / `input` / `voice` / `response_format` / `speed`
  - 扩展字段（声音克隆 / Voice Design）：`prompt_audio` / `prompt_text` / `cfg` / `timesteps` / `seed`
  - 音频响应（wav/mp3）、错误格式对齐 OpenAI
- `runtime-cloud` 协议**单独定义**（Gradio），不并入 shared-proto：
  - 实测官方 `voxcpm.modelbest.cn` 为 **Gradio 6.10**，端点 `POST /gradio_api/predict` 命名 `/generate`
  - 参数：`text` / `control_instruction` / `ref_wav` / `use_prompt_text` / `prompt_text_value` /
    `cfg_value` / `do_normalize` / `denoise` / `dit_steps` / `user_id`（详见 README 云端契约表）
  - 返回 Gradio `FileData`
- ⚠️ 注意：云端非 OpenAI 契约，TS 客户端「切换」= 切换契约适配层，不是仅改 endpoint
- 术语落地到代码注释与 README
- 各 Runtime 的 `RuntimeConfig` 初步结构
- **配置系统**（`packages/config`）：三层加载 —— 代码默认 < `settings.json` < `.env`/环境变量
  - 默认 `model_dir = models/voxcpm2`，含 `burn/` `pytorch/` 子目录
  - `settings.json` 个人差异大 → gitignore，仅提交 `settings.example.json` 模板
  - 秘密最小化，仅用 `.env`（如 `VOXCPM_API_KEY`），不做全量 `.secrets.json`；`.env` 可不存在
  - gitignore：`models/`、`settings.json`、`.env`；入库模板：`settings.example.json`、`.env.example`

**完成标准**：`shared-proto` 编译通过（本地 OpenAI 契约）；云端 Gradio 契约文档化并可由 runtime-cloud 复用；`config` 加载器能从默认+settings+env 正确合并。

## 阶段 M1 — burn 运行时：独立调用
**目标**：把 `voxcpm-rs` 包装成可直接调用的 burn 运行时。

- `runtime-burn` crate，依赖 `submodule/voxcpm-rs`
- 封装 `VoxCPM::from_local` + `generate`，暴露简洁 API：
  `BurnRuntime::load(config)` / `.generate(text, opts)`
- 支持 Device：`cpu`（burn/ndarray）、`vulkan`（burn/wgpu 或 vulkan）
- 注意 voxcpm-rs 的 `[patch.crates-io]`（bf16 修复）需在 workspace 继承
- 最小可运行 demo：给定权重 + 文本 → 输出音频

**完成标准**：`cargo run` 用 cpu/vulkan 各跑通一次推理。

## 阶段 M2 — burn 运行时：独立 HTTP 服务器
**目标**：burn 运行时自己起一个 HTTP 服务，契约对齐云端。

- `runtime-burn` 内置 axum 服务器（或独立 bin）
- 端点复用 `shared-proto`：`POST /v1/infer`、`POST /v1/models/load`、`GET /healthz`
- 支持 device 选择（启动时或请求级）
- 与云端 `voxcpm.modelbest.cn` 同构，可被同一客户端消费

**完成标准**：`curl` 调本地 burn 服务器完成推理，格式与云端一致。

## 阶段 M3 — TS 客户端
**目标**：TS 侧统一访问本地（OpenAI 契约）与云端（Gradio 契约）。

- `ts-client`：统一 `Client`，内部按目标选择契约适配层
  - 本地：OpenAI `/v1/audio/speech`（`endpoint` = `http://127.0.0.1:PORT`）
  - 云端：Gradio `/gradio_api/predict` `/generate`（`endpoint` = `https://voxcpm.modelbest.cn`）
- 音频结果处理（ArrayBuffer / Blob / wav）
- 自动回退：本地不可用 → 云端（可配置，注意契约转换）
- 类型与 `shared-proto` 对齐（本地）；云端 Gradio 类型单独定义

**完成标准**：同一段 TS 代码，通过切换 `Client` 适配层在本地 burn / 云端间工作（非仅改 URL）。

## 阶段 M4 — MVP 软件（CLI）
**目标**：给使用者最小可用的「软件」。

- `cli`：`vox serve --runtime burn --device vulkan` 起本地服务器；
  `vox infer --text "..."` 一键推理
- 与 TS 客户端共用 `shared-proto`
- 快速开始文档（权重 → 推理 → 音频，5 分钟内）

**完成标准**：新用户按 README 跑通本地推理。

## 阶段 M5 — pytorch 运行时
**目标**：第二条独立运行时，覆盖 PyTorch 生态设备。

- `runtime-pytorch` crate：参考 `submodule/VoxCPM`（官方 PyTorch 实现，`torch>=2.5`，
  设备用 `torch.device` cpu/cuda，推理入口在 `src/voxcpm/core.py` 与 `cli.py`）
- 集成方式候选：① `pyo3` 直接调 Python 模型；② 把官方实现包一层 HTTP 服务由 Rust 拉起
- 支持 Device：`cpu` / `cuda`
- 同样暴露 `RuntimeConfig` + 独立调用 + 独立 HTTP 服务器（复用 shared-proto）
- 与 burn 运行时并行，互不知晓（去中心化）

**完成标准**：pytorch 运行时本地服务器与 burn 服务器被同一 TS 客户端消费。

## 阶段 M6 — 扩展 Device
**目标**：补齐更多设备。

- burn：`cuda`（burn/cuda）、`rocm`（burn/rocm）
- pytorch：`rocm`
- 数值对齐测试（同输入同输出近似）

**完成标准**：burn/pytorch 在各 Device 下产出一致音频。

## 阶段 M7 — 多模型扩展
**目标**：支撑「之后支持更多模型」。

- 模型注册表 / 插件机制：按名字取 `Runtime` + 模型实现
- 抽象 `TtsModel` / `VcModel` 等高层 trait
- 文档：如何新增一个运行时或模型

**完成标准**：新增模型仅需实现 trait + 注册，调用方无感。

## 移植 — 从 LocalDub/voxlab 迁入（Rust cloud 链路已完成）

**背景**：本仓库的 Rust 化正在推进（voxcpm-cloud 的 TS 端口已完成，
源码实测自 LocalDub `packages/voxlab`）。LocalDub 侧 `packages/voxlab`
是同一批能力的先行实现（Rust lib + TS 引擎），按职责拆分迁入本仓库。

**映射方案**：

| LocalDub `packages/voxlab` | → 本仓库 | 归属逻辑 |
|---|---|---|
| Rust `wav.rs` (read/write/resample) | `vox-core` | 通用 wav 工具, 与 VoxCPM 无关 |
| Rust `gradio_client.rs` | `vox-core` | 通用 Gradio HTTP 客户端 (任何 Gradio 服务) |
| Rust `engines/voxcpm/cloud.rs` | `voxcpm-cloud` | VoxCPM 专属语义 (cfg/dit_steps/prompt/user_id) |
| Rust `main.rs` (cloud 冒烟 bin) | `voxcpm-cloud` main/example | 冒烟的是 cloud 后端 |
| TS `engines/voxcpm/cloud.ts` | ✅ 已移植 (`voxcpm-cloud/src/index.ts`) | Gradio 6.10 契约实测 |

**已完成**：voxcpm-cloud 的 TS 端口（契约实测 + 与本地 torch server 同签名互换）。

**待办**：
- [x] Rust `cloud.rs` (VoxCPMCloud/Config/TtsResult) 移入 `voxcpm-cloud`
- [x] Rust `wav.rs` 移入 `vox-core`；自研 `gradio_client.rs` 已删除，改用官方推荐的 `gradio` crate（支持 Gradio 4/5/6）
- [x] 冒烟 CLI 移为 `examples/smoke.rs`（clap 派生参数），`main.rs` 删除
- [x] Gradio 客户端曾尝试替换为 `gradio` crate —— **实测失败**: 6.10 的
      `/gradio_api/info` 中 `parameters[].label` 是 i18n map（非 string）,
      `Client::new_sync` 解析即失败。已回退自研客户端（对 6.10 实测可用）。
      已报上游 issue: https://github.com/JacobLinCool/gradio-rs/issues/10
      （含响应样例与修复建议; 上游修复后可用 `examples/verify_config.rs` 验证）
- [ ] LocalDub 侧 `stages/tts/mod.rs` 的依赖切换 (见未决)
- [ ] `packages/voxlab` 在 LocalDub 侧退役

**未决**：
- LocalDub 如何依赖本仓库: path 依赖 / git 依赖 / 保留副本 (取舍:
  单一真相 vs 跨仓库耦合; LocalDub 生产在用, 切换前 vox-lab 需稳定)
- TS 本地引擎 (`download.ts` / `engines/voxcpm/onnx-node` / `pth`) 是否
  迁入: 它们是本地 VoxCPM 推理实验 (非 cloud); LocalDub 的 TS 路径
  (`voxcpm_torch_gradio.ts`) 只用 `VoxCPMCloud`, 疑似死路径待确认

**注意**：LocalDub 侧 cloud TTS 生产路径在用 (`stages/tts/mod.rs`),
切换依赖前保持其可用性。

## 未来候选（未排期）
- WASM / NAPI：TS 直接内嵌本地推理（替代 HTTP，纯前端离线）
- 流式推理（边生成边播放）
- 量化 / 低精度后端
- 集中式调度层（若多个 Runtime 需统一编排再考虑，当前不做）
- 更多模型接入（其他 TTS、VC、声码器）

## 决策记录
- 框架实现叫 **Runtime**，计算后端叫 **Device**（复用 Burn 的 backend 概念）。
- **暂不做调度层**，用 `RuntimeConfig` 替代；先各 Runtime 自包含。
- 每个 Runtime 能独立调用、独立起服务器。
- 以 `submodule/voxcpm-rs`（Burn 实现）作为 burn 运行时的基础依赖/参考。
- 以 `submodule/VoxCPM`（官方 PyTorch 实现）作为 pytorch 运行时的参考。
- **实测**：官方 `voxcpm.modelbest.cn` 是 Gradio 6.10 应用，非 OpenAI 服务。
  云端契约 = Gradio `/gradio_api/predict` `/generate`，由 `runtime-cloud` 单独对接；
  本地运行时仍采用 OpenAI `/v1/audio/speech` 作为自有标准契约。两者不互通。
- 权重范围：仅推理/部署，不含训练与微调。
