# 决策记录（ADR）

本目录记录 vox-lab 的架构与实现决策，一条决策一个文件，按编号递增、只追加。
采用轻量 ADR 格式：每个文件含 frontmatter（number / title / status / date）
与「状态 / 背景 / 决策 / 后果」小节。

## 为什么独立成目录

立项时的决策散落在 `README.md` 设计原则、`ROADMAP.md` 决策记录及各章节的
「未实现 / 实际决策」标注里，多处重复且容易不同步（例如 Rust gradio 客户端
从自研切到 `gradio` crate 后，ROADMAP 仍留着「已回退自研」的旧记录）。
现在以本目录为**唯一真相**：README / ROADMAP 只保留事实性描述 + 指向这里的链接。

## 索引

| 编号 | 决策 | 状态 |
|------|------|------|
| [0001](./0001-terminology-and-scope.md) | 术语（Runtime / Device）与项目范围 | accepted |
| [0002](./0002-no-scheduler-compile-time-device.md) | 不做调度层；Device 由编译期 feature 决定 | accepted |
| [0003](./0003-rust-cli-python-http.md) | Rust 侧是库 + CLI，只有 Python 侧起 HTTP 服务 | accepted |
| [0004](./0004-gradio-generate-contract.md) | 全链路只用一套 Gradio `/generate` 契约 | accepted |
| [0005](./0005-gradio-rs-client.md) | Rust 客户端选用 `gradio` crate v0.4.2（废弃自研） | accepted |
| [0006](./0006-submodules-and-runtimes.md) | 子模块与运行时依赖选型 | accepted |
| [0007](./0007-config-conventions.md) | 配置约定（模板入库、加载器未实现） | accepted |

## 状态取值

- `proposed` — 提出，未定
- `accepted` — 已采用
- `deprecated` — 已被后续决策取代（文件保留，注明取代者）
- `superseded` — 同上，正文写明被哪条取代

## 如何新增一条决策

1. 取下一个编号，文件名 `NNNN-简短英文短横线标题.md`（如 `0008-xxx.md`）。
2. 复制任一现有文件的 frontmatter 与小节结构；`date` 用当天日期。
3. 在「决策」里写清**选了什么、废弃了什么**，在「后果」里列出需要同步的
   README / ROADMAP 位置与代码位置。
4. 在本索引表加一行。
5. 若取代了旧决策，把旧文件 `status` 改为 `deprecated` / `superseded` 并互相链接，
   同时更新 README / ROADMAP 中受影响的描述——不要只改一处。

## 与「历史规划」的边界

`ROADMAP.md` 的 M0–M7 是**立项时的阶段规划**，不是决策。它们保留作历史记录，
已标 ⛔ 并链到本目录；新决策写在这里，不回头改写那些阶段正文。
