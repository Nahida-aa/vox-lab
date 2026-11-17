# Burn GPU 后端探索（voxcpm-rs）

记录在 780M 上把 VoxCPM2 跑进 Burn（wgpu / Vulkan）的过程、各类失败与最终能用的组合。
结论索引在 [README.md](./README.md)，决策在 [ADR 0011](../../decisions/0011-voxcpm-gpu-backend-outlook.md)。

## 环境

- Radeon 780M（gfx1103，RADV，Mesa 26.x）
- 可见 VRAM：`/sys/class/drm/card*/device/mem_info_vram_total` = **4 GB**；
  其余走 GTT（共享内存），启动参数未设 `amdgpu.vramlimit`/`gttsize`
- 内核 zen，32 GB 物理内存（实验时已用约 16 GB）
- `voxcpm-rs` 独立 workspace（`[patch.crates-io]` 指向 `vendor/patched/*`）

## 失败与原因（按遇到顺序）

| # | 现象 | 原因 | 处理 |
|---|------|------|------|
| 1 | `cubek-matmul` panic：`BF16 lhs/rhs/output not supported` | gfx1103 无 bf16 matmul；`Cube` 策略直接 panic 而非回退 | 见下方「三条绕法」 |
| 2 | `wgpu`(f32)：`radv/amdgpu: Not enough memory for command submission` | bf16 权重 4.4 GB → f32 8.7 GB，超出 4 GB VRAM + GTT 可用余量 | 改用 f16（同 2 字节） |
| 3 | `vulkan`(f32) 同一错误 | 同上 | 同上 |
| 4 | 同一文本多次生成长度不同（0.64 / 2.88 / 2.56 s） | CFM 初始噪声与 VAE `NoiseBlock` 都无种子 | 加 `GenerateOptions::seed` |

## 绕开 bf16 matmul 的三条路

### 1. 换浮点类型（已采用，可用）

`cli` 的 vulkan/wgpu 二进制按 `VOXLAB_FLOAT=f16|f32|bf16` 选择后端类型。
f16 与 bf16 同为 2 字节，**容量不变**，但 f16 的动态范围只有 ±65504。

| 组合 | 结果 |
|------|------|
| Vulkan + f16 | **RTF 3.9–5.6，稳定可用** |
| wgpu + f16 | RTF 5.66，与 Vulkan f16 的相关性 0.9311（采样随机导致，非实现差异） |
| Vulkan + bf16 | panic（无 matmul） |
| wgpu/vulkan + f32 | OOM |

### 2. 强制朴素 matmul（已验证可行，但太慢）

`vendor/patched/burn-cubecl` 新增 `MatmulStrategy::Naive` 与全局开关
`set_force_naive_matmul(true)`（CLI 侧 `VOXLAB_NAIVE_MATMUL=1`）。`Naive` 标量实现对
任何 float 类型都成立，因此 **bf16 能跑通**：

```
matmul: naive (VOXLAB_NAIVE_MATMUL)
Got 307200 samples (6.40s audio) in 490.09s (RTF=76.58)
```

可跑但无实用价值（比 f16 的 4.2 慢 18 倍）。它的价值是验证「bf16 数值链路本身没问题」。

### 3. 等驱动暴露 bf16（未验证）

`cubecl-spirv` 的补丁里已有 `Bf16PromoteTransform`：把不支持的 bf16 指令提升到
f32 计算再转回，注释写明 RADV 自 Mesa 25.2 起只强制支持 bf16 的
`OpFMul`/`OpFMulAdd`/`OpDot`。本机 `vulkaninfo` 没有 `VK_KHR_shader_bfloat16`，
因此这条路径当前不可用。**是否会被 RADV 支持属于推断，未查到明确计划。**

## 结论

- **能用**：Vulkan + f16（RTF 3.9–5.6）、wgpu + f16（RTF 5.66），均比 PyTorch CPU（8.29）快。
- **不能用**：bf16（缺 matmul）、f32（显存不足）、Burn CPU（RTF 338）。
- f16 的精度风险尚未量化：本次只验证了「能跑」，没有与 PyTorch bf16 做 SNR 对比。
