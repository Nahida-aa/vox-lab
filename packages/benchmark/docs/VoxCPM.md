## VoxCPM Benchmark — PyTorch vs Rust tch

**Hardware**: AMD Ryzen (CPU), DDR5 ~40 GB/s bandwidth  
**Date**: 2026-06-28  
**Model**: OpenBMB/VoxCPM2 (bf16)  
**Params**: timesteps=10, cfg=2.0, max_len=500

| text | PyTorch gen | PyTorch RTF | tch gen | tch RTF |
|------|------------|-------------|---------|---------|
| short (8字) | 10.81s / 1.76s | **6.14** | 13.16s / 1.12s | **11.75** |
| medium (45字) | 35.70s / 8.48s | **4.21** | 88.66s / 8.00s | **11.08** |

**PyTorch faster by 2–2.5×**. Root cause: Burn/LibTorch kernel integration less optimized than PyTorch oneDNN on CPU. No point testing tch f32 — bf16 (half memory bandwidth) is already slower.
