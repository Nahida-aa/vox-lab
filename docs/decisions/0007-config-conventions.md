---
number: 7
title: 配置约定（模板入库、加载器未实现）
status: accepted
date: 2026-11-14
---

# 7. 配置约定（模板入库、加载器未实现）

## 状态

模板约定已接受；三层加载器未实现。

## 背景

立项设计了三层配置，优先级从低到高：
**代码默认 < `settings.json` < `.env` / 环境变量 / 秘密**。
规划由 `packages/config` 加载器完成合并，并由 `RuntimeConfig`（见
[0002](./0002-no-scheduler-compile-time-device.md)）承载选择。

## 决策

- **入库模板**（已落地）：根目录 `settings.example.json`、`.env.example`，
  均不含真实值。个人配置 `settings.json` / `.env` 经 `.gitignore` 忽略，
  `.env` 可不存在（纯本地推理无需秘密）。
- **秘密最小化**：仅用 `.env` 承载敏感 / 需覆盖的键值对，不引入全量 `.secrets.json`。
  预留项如 `VOXCPM_API_KEY`、`VOXCPM_CLOUD_API_URL`、`VOXCPM_MODEL_DIR`。
- **加载器未落地**：`packages/config` crate、`RuntimeConfig` 类型、
  `settings.json` 三层合并均**不存在**。README「配置系统」节的 JSON 示例是立项约定，
  保留备查，不代表已实现。
- **实际配置入口**：各 CLI 的 clap 参数、Python 服务的命令行参数与环境变量
  （如 `DEMUCS_MODEL_DIR`）、`voxcpm-cloud` 的 `VoxCPMCloudConfig`。

## 后果

- README「配置系统」节已标注「约定，未实现加载器」。
- 若未来实现加载器，应作为新 ADR 记录，并同步更新
  [0002](./0002-no-scheduler-compile-time-device.md) 中「无 RuntimeConfig」的表述。
