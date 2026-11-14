---
number: 1
title: 术语（Runtime / Device）与项目范围
status: accepted
date: 2026-11-14
---

# 1. 术语（Runtime / Device）与项目范围

## 状态

已接受（部分为立项约定，范围已扩展）。

## 背景

立项时以「voxcpm2 多后端推理」为目标，需要先固定术语，避免把不同层次的概念混用。

## 决策

- **Runtime（运行时）**：一套推理实现，如 `burn` / `pytorch` / `cloud`。用户选的是「用哪套实现来跑」。
- **Device（设备）**：计算后端 / 硬件目标，如 `cpu` / `vulkan` / `cuda` / `rocm` / `metal`。
  Burn 内部称之为 `Backend`，本项目对外统一称 Device，内部映射即可，不冲突。
- **只做推理 / 部署**：假设已有模型权重，不含训练与微调。

> 注：早期设想的 `RuntimeConfig`（运行时的参数集合，承载权重路径 / device / 并发 / 采样，
> 「配置即调度」）**未落地**，见 [0002](./0002-no-scheduler-compile-time-device.md)。

## 范围演变

仓库最初只覆盖 VoxCPM TTS / VC。随着 LocalDub 相关能力迁入，现扩展为
**ASR（whisper）+ 音源分离（demucs）+ TTS / VC（VoxCPM、CosyVoice）** 的语音全链路
实验与 Rust 化工作区，另含基准测试（`packages/benchmark/`）、研究记录（`packages/research/`）
与 Linux / Windows 预构建资产发布（`.github/workflows/release-*.yml`）。

## 后果

- README 的「术语」表保留 Runtime / Device 定义；`RuntimeConfig` 一行标注为未落地。
- README / ROADMAP 的 M0–M7 阶段划分是立项时以纯 VoxCPM 为范围的规划视角，
  保留作历史记录，不按原命名推进。
