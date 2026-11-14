---
number: 5
title: Rust 客户端选用 gradio crate v0.4.2（废弃自研客户端）
status: accepted
date: 2026-11-14
---

# 5. Rust 客户端选用 `gradio` crate v0.4.2（废弃自研客户端）

## 状态

已接受。用户是该上游 crate 的维护者之一。

## 背景

`voxcpm-cloud` 的 Rust 侧需要按 [0004](./0004-gradio-generate-contract.md) 的
Gradio `/generate` 契约对接云端。早期曾用自研 `gradio_client.rs`（极简 Gradio 5
HTTP 客户端，`/gradio_api/call/{api}` 两段式 + SSE）。

首次尝试替换为 `gradio` crate 时**实测失败**：Gradio 6.10 的
`/gradio_api/info` 中 `parameters[].label` 是 i18n map（非 string），
`Client::new_sync` 解析即失败，因此一度回退自研客户端，并报上游
https://github.com/JacobLinCool/gradio-rs/issues/10（含响应样例与修复建议）。

## 决策

- **上游已修复**该 6.10 label i18n 兼容问题。现依赖 pinned git tag：
  ```toml
  # packages/voxcpm-cloud/Cargo.toml
  gradio = { git = "https://github.com/JacobLinCool/gradio-rs", tag = "v0.4.2" }
  ```
  `Cargo.lock` 锁定 `0.4.2` / commit `4e1fff1c2903c79ca702a969b1acf726ddecd842`。
- 支持 Gradio 4 / 5 / 6。`packages/voxcpm-cloud/src/cloud.rs` 直接使用
  `gradio::{Client, ClientOptions, PredictionInput, PredictionOutput}` 调用 `/generate`。
- **自研 `gradio_client.rs` 已删除、不再保留**；也不迁入 `vox-core`。
- `packages/voxcpm-cloud/examples/verify_config.rs` 保留作 `/config` 与
  `/gradio_api/info` 响应校验参考（其中类型定义从 crate 源码复制，因 crate 未导出）。

## 后果

- `vox-core/src/lib.rs` 里声明不存在模块的 `gradio_client` 文档注释已清除。
- ROADMAP 移植章节的「实测失败、已回退自研」记录已更新为「上游已修复、现用 v0.4.2」。
- 依赖是 git tag 而非 crates.io 版本：升级时需同步核对 `Cargo.lock` 与 6.10 兼容性。
