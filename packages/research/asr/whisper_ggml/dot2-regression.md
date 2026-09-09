# fp16: 1 → dot2 性能回归

## 现象

旧 `whisper-vulkan`（v1.8.x, 6月9日构建）GPU 23.3s (RTF 0.137)
新 `whisper-cli`（v1.9.1,  6月30日构建）GPU 31.3s (RTF 0.184)

旧二进制 **快 34%**。Vulkan 设备信息唯一区别：

- 旧: `fp16: 1`
- 新: `fp16: dot2`

## 根因

上游 commit `686bc802`（6月15日）新增 `v_dot2_f32_f16` 支持：

```
vulkan: add `v_dot2_f32_f16` support in matrix-matrix multiplication
        and Flash Attention (llama/24123)
```

`v_dot2_f32_f16` 是 RDNA3 原生 VOP3P 指令，理论上一次指令完成 2 组 fp16→f32 点积累加，比非 dot2 路径（4 次标量 FMA）更高效。

**但** `vulkan-shaders-gen.cpp:339` 中，dot2 shader **跳过了 `spirv-opt` 优化**：

```cpp
// disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize
// SPV_VALVE_mixed_float_dot_product capability)
if (!coopmat && name.find("bf16") == std::string::npos
    && name.find("rope") == std::string::npos
    && name.find("_dot2") == std::string::npos) {
```

- 非 dot2 shader → `spirv-opt` 优化（死代码消除、指令合并等）→ 高效 SPIR-V
- dot2 shader → 原始未优化 SPIR-V → 周围代码质量差，净效果变慢

**结论：`v_dot2_f32_f16` 指令本身更快，但因 SPIR-V 工具链不支持 `SPV_VALVE_mixed_float_dot_product` 能力集，整个 shader 跳过优化，性能反而不如非 dot2 路径。**

## 系统环境

- GPU: AMD Radeon 780M (RDNA3, Phoenix)
- 驱动: RADV (Mesa)
- 两者均检测到 `KHR_coopmat` matrix cores

## 解决方向

1. **上游修复**: 等 `spirv-opt` 支持 `SPV_VALVE_mixed_float_dot_product`
2. ~~本地打补丁~~: 上游代码第 5604、6415 行已内置 `GGML_VK_DISABLE_DOT2`，无需 patch
3. **环境变量（已采用）**: `GGML_VK_DISABLE_DOT2=1` 跳过 `VK_VALVE_shader_mixed_float_dot_product` 扩展检测，回退到 `fp16: 1` 路径
   - `whisper_ggml_cli_vulkan.ts`: `spawnSync` 传 `env: { ...process.env, GGML_VK_DISABLE_DOT2: '1' }`
   - `run-whisper-cli.ts`: 同上
4. **不动**: 性能差异只影响 whisper.cpp 推理阶段，用户若不在意可跳过
