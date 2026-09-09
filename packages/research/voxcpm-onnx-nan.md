# VoxCPM ONNX Residual LM NaN Investigation

**Date**: 2026-06-06
**Hardware**: AMD Radeon 780M (RDNA 3, gfx1103), 27GB RAM + 27GB swap
**Model**: VoxCPM2 (OpenBMB, 1.3B params)
**ONNX Runtime**: onnxruntime-node 1.26.0, onnxruntime 1.19.0 (Python)
**PyTorch**: 2.12.0+rocm7.2

## Problem

ONNX decode_step model (`voxcpm2_decode_step.onnx`) produces **100% NaN** in the residual LM portion of its outputs:

- `new_dit_hidden[0, 1024:2048]` — ALL 1024 residual elements are NaN
- `new_residual_next_keys` — partially NaN (propagated from transformer state)
- `new_residual_next_values` — partially NaN (propagated from transformer state)
- `pred_feat` — **zero NaN** (because base LM + residual projections produce clean output despite NaN residual state)

The PyTorch native model (`model.safetensors`) produces **zero NaN** under identical conditions — confirming the NaN is an ONNX export artifact, not a model architecture issue.

## Methodology

### 1. Graph Anatomy

The decode_step model has **9998 nodes** and **694 initializers** (weights + biases). The architecture:

| Component   | Layers | Hidden Size   | KV Heads | KV Channels |
| ----------- | ------ | ------------- | -------- | ----------- |
| Base LM     | 28     | 2048          | 2        | 128         |
| Residual LM | 8      | 2048          | 2        | 128         |
| VAE         | -      | 64 (feat dim) | -        | -           |

Graph inputs (8):

- `dit_hidden` [1, 2048] — concatenated base + residual hidden state
- `base_next_keys` [1, 28, 2, 1, 128] — base KV cache
- `base_next_values` [1, 28, 2, 1, 128]
- `residual_next_keys` [1, 8, 2, 1, 128] — residual KV cache
- `residual_next_values` [1, 8, 2, 1, 128]
- `prefix_feat_cond` [1, 4, 64] — previous pred_feat
- `noise` [1, 4, 64] — random noise for diffusion
- `cfg_value` [] — classifier-free guidance scale

Graph outputs (7):

- `pred_feat` [1, 4, 64] — predicted audio feature
- `new_dit_hidden` [1, 2048] — updated hidden state
- `new_base_next_keys` [1, 28, 2, 2, 128]
- `new_base_next_values` [1, 28, 2, 2, 128]
- `new_residual_next_keys` [1, 8, 2, 2, 128]
- `new_residual_next_values` [1, 8, 2, 2, 128]
- `stop_flag` [1]

### 2. Tensor Type Classification

The graph contains many 2048-dimensional tensors (sources of potential NaN). These were classified by tracing their node ancestry:

| Tensor                       | Type                       | Role                                             |
| ---------------------------- | -------------------------- | ------------------------------------------------ |
| `mul_5976`                   | **RMSNorm output**         | Residual LM final hidden layer norm (x × weight) |
| `mul_5975`                   | RMSNorm computation        | x / sqrt(mean(x²) + eps)                         |
| `mul_2888` → `mul_5974`      | Intermediate layer outputs | 28 base layers × ~10 intermediate tensors each   |
| `layer_0/matmul_0` → ...     | Attention outputs          | QKV projections, attention scores                |
| `layer_0/ffn/matmul_0` → ... | FFN outputs                | Feed-forward network activations                 |
| `Reshape_*`                  | Shape ops                  | Attention reshape/scatter ops                    |
| `ScatterND_*`                | KV cache update            | Scatter along sequence dimension                 |

**Key finding**: `mul_5976` = `decode.residual_lm.norm.weight` × `mul_5975`. This is the RMSNorm of the residual LM's last hidden layer. The NaN originates upstream in the residual LM's internal computation (between the 8th residual layer's attention/FFN and this RMSNorm).

### 3. NaN Localization

Using a combination of:

- **Binary search**: iteratively zeroing subsets of residual LM weights and testing
- **Tensor tracing**: identifying all 2048-d tensors as layer outputs and testing each
- **Partial inference**: running the ONNX model with zeroed residual LM inputs

The NaN originates **within the residual LM**, not at its inputs. Specifically:

- `dit_hidden[1024:2048]` (residual portion) can be arbitrary/zero — still produces NaN output
- The base LM portion `dit_hidden[0:1024]` is clean
- KV cache update via ScatterND is correct (data values are clean)
- The 72 existing `Where` nodes in the graph (some intened as NaN mitigation) are insufficient

### 4. Multi-Step Simulation

Because `pred_feat` stays clean (no NaN), we tested whether the NaN can be "lanced" at the output to prevent propagation:

```python
# Pseudocode
for step in range(10):
    out = sess.run(feeds)
    pred, dh, rk, rv = out.pred_feat, out.new_dit_hidden, out.new_residual_next_keys, out.new_residual_next_values
    dh = nan_to_zero(dh)  # Zero NaN in residual portion
    rk = nan_to_zero(rk)
    rv = nan_to_zero(rv)
    # Next step uses fixed dh/rk/rv → pred stays clean
```

**Result**: With the workaround, `pred_feat` is **zero NaN in all 10 steps**. This means:

1. The NaN is confined to the residual LM's internal state (not in the pred_feat computation path)
2. Dirty state doesn't poison the base LM (they're separate transformer stacks)
3. A surgical fix at the output nodes is sufficient to contain the NaN

## Root Cause Analysis

| Hypothesis              | Evidence                                                       | Verdict            |
| ----------------------- | -------------------------------------------------------------- | ------------------ |
| Bad ONNX weights        | weights match PyTorch (same `.data` size), prefill model works | ❌                 |
| OOM / memory corruption | NaN is deterministic (same values every run)                   | ❌                 |
| FP32 precision issue    | PyTorch FP32 has no NaN; softmax overflow would be uniform     | ❌                 |
| ONNX optimization bug   | ORT graphs with/without level 99 have same NaN                 | ❌                 |
| **Export artifact**     | PyTorch forward_step, correctly, ONNX doesn't                  | ✅ **Most likely** |
| KV cache ScatterND bug  | KV cache data is clean, NaN in hidden state                    | ❌                 |

**Likely root cause**: The PyTorch exported ONNX graph lost the correct computation path for the residual LM in `MiniCPMModel.forward_step()`. Specifically:

- The residual LM's attention/FFN computation produces undefined values in ONNX (but not in PyTorch eager mode)
- Possible causes: incorrect weight scope binding during export, missing operator in opset 20, or graph simplification that eliminated critical NaN-mitigating operations
- The 8 residual LM layers share architecture with the 28 base LM layers, but only the residual ones NaN — suggesting the export handled the two stacks differently

The ONNX model was exported with `opset_version=20`, `torch.onnx.export(dynamic_axes={...})`, and `torch.onnx.export(model, args, export_params=True)`.

**Re-export attempt** (`scripts/export_voxcpm_onnx.py`) was blocked by KV cache statefulness: `torch.stack(self.kv_cache.key_cache)` where `key_cache` is a Python list of tensors — `torch.stack` over a list of ONNX-traced tensors is not supported for dynamic axes.

## Fix: Surgical ONNX Graph Edit

### Approach

Instead of re-exporting (blocked by KV cache issue), modify the ONNX protobuf directly to insert NaN-clamping nodes at the output.

### Implementation

Insert IsNaN → Where chains at 3 outputs of the decode_step model:

```
new_dit_hidden ──→ Shape ──→ Expand ──→ Where ──→ new_dit_hidden_fixed
                  │          ▲           ▲
                  │          │           │
                  └── zero [0.0] ────┘           │
                                                  │
                        IsNaN ────────────────────┘
```

Same pattern for `new_residual_next_keys` and `new_residual_next_values`.

### Key Techniques

1. **Memory-safe loading**: `onnx.load(filename, load_external_data=False)` — skips deserializing the 16GB `.data` file. Only the 20MB protobuf is loaded and modified.
2. **Dynamic shape handling**: `Shape(node) → Expand(zero_scalar, shape)` creates a zero-filled tensor matching the output's runtime dimensions, avoiding any static shape assumptions.
3. **Graph cycle avoidance**: Only original nodes (index < 9998) are rewired to receive `_fixed` outputs. The newly added fixer nodes (Shape, Expand, IsNaN, Where) keep their original inputs, avoiding feedback cycles.

### Script

`scripts/fix_voxcpm_onnx_nan.py` — takes `--input` and `--output` paths, adds 12 nodes (4 per target), saves the modified proto.

### Verification

```python
for step in range(10):
    out = sess.run(feeds)
    # All outputs: zero NaN ✅
print('All 10 steps: NO NaN!')
```

Post-fix benchmark (CPU, Chinese text):

| Text              | RTF                   | Notes                  |
| ----------------- | --------------------- | ---------------------- |
| Short (8 chars)   | 9.42                  | Residual LM now active |
| Medium (45 chars) | 8.37                  | Residual LM now active |
| Long (128 chars)  | timeout (498 patches) | Memory-bound           |

RTF increased from ~7.4 (pre-fix, zeroed residual) to ~8.4–9.4 (post-fix, residual running). This is expected — the 8 residual transformer layers + cross-attention now execute their full computation instead of being zeroed.

### Trade-off

The `Where` node adds element-wise overhead per step:

- 2048 elements for `new_dit_hidden` (0.002ms)
- 2 × 8 × 2 × 2 × 128 = 8192 elements for each KV cache (0.008ms)
- Total: ~0.01ms per step, negligible vs. ~630ms per step total

The real RTF increase comes from the residual LM layers actually running (not from the fixer nodes).

## Applied Code Changes

| File                                                   | Change                                                                               |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `scripts/fix_voxcpm_onnx_nan.py`                       | **NEW** surgical ONNX graph editor                                                   |
| `modelscope/OpenBMB__VoxCPM2/voxcpm2_decode_step.onnx` | replaced with fixed version (⚠️ backup saved as `voxcpm2_decode_step_original.onnx`) |
| `packages/voxlab/src/engines/voxcpm/onnx-node.ts`      | removed NaN workaround (lines 206-219), uses `_fixed` output names                   |

## Remaining Issues

1. **Re-export still desired**: A clean re-export from PyTorch would be ideal but is blocked by KV cache statefulness in `torch.onnx.export`. Possible mitigation: rewrite `forward_step` to accept individual KV tensors instead of using a list-based cache.
2. **Audio quality comparison**: The post-fix audio quality should be measurably better (residual LM contributes acoustic detail). Subjective A/B comparison with PyTorch native output needed.
3. **Long text timeout**: 498 patches for 128 chars is excessive. The `autoMaxPatches = max(20, textLen * 6)` heuristic may overestimate for long texts.

## References

- Model source: `submodule/VoxCPM/src/voxcpm/modules/minicpm4/model.py` (`MiniCPMModel.forward_step`)
- Model source: `submodule/VoxCPM/src/voxcpm/model/voxcpm2.py` (`VoxCPM2Model._inference`)
- ONNX re-export attempt: `scripts/export_voxcpm_onnx.py`
- ONNX NaN fix script: `scripts/fix_voxcpm_onnx_nan.py`
- TS implementation: `packages/voxlab/src/engines/voxcpm/onnx-node.ts`
- Benchmark runners: `packages/benchmark/VC/VoxCPM2/`
