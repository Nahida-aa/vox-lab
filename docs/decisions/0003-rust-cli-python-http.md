---
number: 3
title: Rust 侧是库 + CLI，只有 Python 侧起 HTTP 服务
status: accepted
date: 2026-11-14
---

# 3. Rust 侧是库 + CLI，只有 Python 侧起 HTTP 服务

## 状态

已接受（取代立项「每个 Runtime 能独立起 HTTP 服务器」）。

## 背景

立项设想每个 Runtime 自包含、能独立启动一个 HTTP 服务器，
规划过 `runtime-burn` 内置 axum 服务器、`/v1/infer`、`/healthz` 等端点。

## 决策

- **Rust 侧**：公共逻辑库 + 每个后端一个独立 CLI 二进制（薄壳）。
  - `voxcpm-burn` → `voxcpm-burn-{cpu,vulkan,wgpu,tch}`
  - `demucs-burn` → `demucs-burn-{cpu,vulkan,wgpu,tch,cuda,rocm}`
  - `voxcpm-cloud`、`vox-core`、`voice-cloning`、`util` 均为库。
  - **Rust 侧不起 HTTP 服务**，没有 axum 服务器，`/v1/infer` 等端点不存在。
- **Python 侧**（uv workspace）才起 HTTP 服务：
  - `packages/voxcpm_torch_server`：FastAPI + Gradio（复刻云端 `/generate`），
    另有 `/status`、`/load-model`、`/unload-model` 与 mDNS 发现，默认端口 19112。
  - `packages/demucs_pytorch`：PyTorch 参考实现（校验 `demucs-tch`）。
    （原 `demucs_torch_server` HTTP 服务已由此取代。）
  - `packages/servers_py`：上述服务公共依赖。

## 后果

- README 架构图明确「服务器指 Python 侧」「Rust 侧 crate 是库」。
- ROADMAP M1 / M2 / M5 的「独立 HTTP 服务器」描述保留作历史规划，已标 ⛔。
- 命令行入口为各 CLI 二进制与 Python server 的 `main()`；
  立项规划的 `vox serve` / `vox infer` CLI 未建立。
