# Benchmark

各模型在不同运行环境下的性能基准测试。

## 目录结构

```
packages/benchmark/
├── readme.md
├── VC/           # Voice Cloning（语音克隆）
│   ├── VoxCPM2/
│   ├── CosyVoice2/
│   └── CosyVoice3/
├── 语音分离/      # 未来扩展
└── 语音识别/      # 未来扩展
```

## 文件命名约定

```
<前端>-<后端>.<ext>
```

| 段 | 取值 | 说明 |
|---|------|------|
| 前端 | `PyTorch`, `nodejs-ONNX` | 推理框架 |
| 后端 | `cpu`, `cuda`, `rocm`, `vulkan`, `webgpu` | 执行设备 |
| 后缀 | `.py`, `.ts` | 语言 |

### 示例

| 文件 | 含义 |
|------|------|
| `PyTorch-cpu.py` | PyTorch CPU |
| `nodejs-ONNX-webgpu.ts` | ONNX Runtime Node.js WebGPU |
| `*‑test‑*` | 辅助/兼容性测试，非完整 benchmark |

## 结果文件命名

```
<engine-abbr>-<backend>.json
```

| engine-abbr | 说明 |
|---|---|
| `py` | Python PyTorch |
| `ts` | TypeScript ONNX Runtime |

示例：`py-cpu.json`, `ts-webgpu.json`, `ts-cpu.json`

## 输出格式

每个脚本在 `results/` 下输出 JSON：

```json
{
  "engine": "python | typescript",
  "device": "cpu | cuda | webgpu | ...",
  "generate_time_s": 5.8,
  "output_duration_s": 1.12,
  ...
}
```
