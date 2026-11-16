"""voxcpm_pytorch — VoxCPM 的 PyTorch 实验与本地服务。

模块划分（与 demucs_pytorch 对齐）：

- `core`   模型加载 / 生成 / 计时 / 精度实验（无 HTTP、无 Gradio 依赖）
- `cli`    实验入口：generate / dtype / profile
- `server` 复刻云端签名的 Gradio + FastAPI 服务

实验目的见 docs/decisions/0011-voxcpm-gpu-backend-outlook.md：
先量出 LM 解码 / DiT·CFM / AudioVAE 各自的占比和 bf16 的精度代价，
再决定是否值得做自研 GPU kernel。
"""

from .core import (  # noqa: F401
    DEFAULT_MODEL_DIR,
    generate,
    load_model,
    profiled_generate,
    stage_breakdown,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "generate",
    "load_model",
    "profiled_generate",
    "stage_breakdown",
]
