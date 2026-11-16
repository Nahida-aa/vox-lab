---
number: 4
title: 全链路只用一套 Gradio /generate 契约
status: accepted
date: 2026-11-14
---

# 4. 全链路只用一套 Gradio `/generate` 契约

## 状态

已接受（早期「OpenAI 本地 + Gradio 云端两套并存」设想已废弃）。

## 背景

立项时画过一版架构图：用户 / TS 经「统一 HTTP 契约 = OpenAI `POST /v1/audio/speech`」
访问本地 burn / pytorch 服务器，并认为与云端同构；TS 侧靠「契约适配层」切换本地与云端。

**实测推翻了这个前提**：官方 `https://voxcpm.modelbest.cn` **不是** OpenAI / vLLM-Omni 服务，
而是一个 **Gradio 6.10 应用**。

## 决策

- **废弃**：本地 `POST /v1/audio/speech`、`shared-proto`、「两套契约并存 + 适配层切换」、
  「本地与云端不互通」等说法。代码中不存在 `/v1/audio/speech`。
- **实际采用**：全链路只有一套契约 —— **Gradio `/generate`**。
  - 云端 `voxcpm.modelbest.cn`：Gradio 6.10，`predict('/generate', [...10 参数])`。
  - 本地 `packages/voxcpm_pytorch/server.py` 的 `generate()` **复刻同一签名**
    （`text` / `control_instruction` / `reference_audio` / `ultimate` / `prompt_text` /
    `cfg_value` / `normalize` / `ref_denoise` / `dit_steps` / `_`），
    返回 `(48000, np.ndarray)`，通过 `gr.Interface` + `mount_gradio_app` 暴露。
  - 客户端 `packages/voxcpm-cloud` 按同一协议对接，
    `--target local|cloud` **只换 endpoint**，参数与返回解析一致，无需契约转换。
- `/generate` 位置参数顺序（以 `cloud.ts` 实测为准，优先级高于 `/gradio_api/info` 字段名）：
  ```
  [text, controlInstruction, refFile, isUltimate, promptText, cfg, normalize, denoise, dit_steps, user_id]
  ```
  对应：文本 / Voice Design 指令 / 参考音频(FileData) / 终极克隆开关 / 精准克隆转写 /
  cfg(1–3, 默认 2) / 归一化 / 降噪 / CFM 步数(1–50, 默认 10) / 用户 ID。
- 云端返回 Gradio `FileData`（`result.data[0].url` 即音频，`fetch` 后解析 WAV，默认 48kHz）。
- **不推荐手搓 `/gradio_api/predict` HTTP**；TS 用 `@gradio/client`（参考音频用
  `handle_file(localPath)` 由 SDK 自动上传），Rust 侧选型见
  [0005](./0005-gradio-rs-client.md)。

## 后果

- README「架构总览」「HTTP 契约」按一套 Gradio 契约重写；设计原则第 4 条改为「对齐云端 Gradio 契约」。
- ROADMAP M0 / M2 / M3 的 OpenAI 契约与适配层描述保留作历史规划，已标 ⛔。
- 里程碑表注、当前状态、参考实现结论段均已同步；避免再出现「不互通 / 需适配层」的反向描述。
