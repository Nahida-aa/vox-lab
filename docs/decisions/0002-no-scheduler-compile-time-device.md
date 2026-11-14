---
number: 2
title: 不做调度层；Device 由编译期 feature 决定
status: accepted
date: 2026-11-14
---

# 2. 不做调度层；Device 由编译期 feature 决定

## 状态

已接受（立项「无调度层」成立；「RuntimeConfig 配置即调度」已废弃）。

## 背景

立项设想：各 Runtime 自包含、互不编排，不做集中式「调度中心」，
由 `RuntimeConfig` 承载「选哪个运行时、哪个设备」，即「配置即调度」，
并认为 **Runtime 与 Device 正交**（同一运行时可在运行时切换 device）。

## 决策

- **不做统一调度层** —— 这条保留。各实现可独立调用、互不编排。
- **`RuntimeConfig` 类型与「配置即调度」未实现、已废弃**。代码中不存在该类型。
- **Device 不在运行时切换，而在编译期由 Cargo feature 选定**：
  - `voxcpm-burn`：`cpu` / `wgpu` / `vulkan` / `tch`
  - `demucs-burn`：上述 + `cuda` / `rocm`
  - 一个二进制绑定一个 Burn backend，由 `required-features` 门控。
- 因此「Runtime 与 Device 运行时正交」**不成立**。
- 实际配置入口：各 CLI 的 clap 参数、Python 服务的命令行参数与环境变量
  （如 `DEMUCS_MODEL_DIR`）。`packages/config` 加载器与 `settings.json` 三层合并未落地，
  见 [0007](./0007-config-conventions.md)。

## 后果

- README「设计原则」按实际落地形态重写；「仓库布局」注与术语表 `RuntimeConfig` 行标注未落地。
- ROADMAP M0–M5 中基于该设想的阶段描述保留作历史规划，已标 ⛔。
- 新增后端 = 新增一个 feature + 一个薄壳二进制 + 对应基准与发布资产，而非运行时配置项。
