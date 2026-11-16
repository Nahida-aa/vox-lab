//! Convolutional encoder/decoder layers (tch-rs implementation).
//!
//! Ported from `demucs-core/src/model/conv.rs` (Burn). Each module holds raw
//! `tch::Tensor` weights in PyTorch layout and applies functional ops.

use tch::{Kind, Tensor};

use crate::{DCONV_COMP, DCONV_DEPTH, KERNEL_SIZE, STRIDE};

fn gelu(x: &Tensor) -> Tensor {
    x.gelu("none")
}

fn glu(x: &Tensor, dim: i64) -> Tensor {
    x.glu(dim)
}

/// Learned per-channel scaling (`LayerScale`).
pub struct LayerScale {
    pub scale: Tensor,
}

impl LayerScale {
    /// For conv feature maps `[B, C, ...]`: broadcast as `[1, C, 1]` (dim 1).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x * &self.scale.unsqueeze(0).unsqueeze(2)
    }

    /// For transformer sequences `[B, S, D]`: broadcast as `[1, 1, D]` (dim 2).
    pub fn forward_last(&self, x: &Tensor) -> Tensor {
        x * &self.scale.unsqueeze(0).unsqueeze(0)
    }
}

/// A single residual dilated-conv layer.
pub struct DConvLayer {
    pub conv1_weight: Tensor,
    pub conv1_bias: Option<Tensor>,
    pub norm1_weight: Option<Tensor>,
    pub norm1_bias: Option<Tensor>,
    pub conv2_weight: Tensor,
    pub conv2_bias: Option<Tensor>,
    pub norm2_weight: Option<Tensor>,
    pub norm2_bias: Option<Tensor>,
    pub scale: Tensor,
    pub dilation: i64,
    pub groups: i64,
}

impl DConvLayer {
    /// ch: channel count; dilation = 1 << j.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let residual = x;
        let t = x.conv1d(
            &self.conv1_weight,
            self.conv1_bias.as_ref(),
            [1],
            [self.dilation],
            [self.dilation],
            self.groups,
        );
        let t = t.group_norm(
            1,
            self.norm1_weight.as_ref(),
            self.norm1_bias.as_ref(),
            1e-5,
            false,
        );
        let t = gelu(&t);
        let t = t.conv1d(
            &self.conv2_weight,
            self.conv2_bias.as_ref(),
            [1],
            [0],
            [1],
            1,
        );
        let t = t.group_norm(
            1,
            self.norm2_weight.as_ref(),
            self.norm2_bias.as_ref(),
            1e-5,
            false,
        );
        let t = glu(&t, 1);
        let ls = LayerScale {
            scale: self.scale.shallow_clone(),
        };
        let t = ls.forward(&t);
        let out_t = t.size()[2];
        residual.narrow(2, 0, out_t) + t
    }
}

/// Stack of `DCONV_DEPTH` residual dilated-conv layers.
pub struct DConv {
    pub layers: Vec<DConvLayer>,
}

impl DConv {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut t = x.shallow_clone();
        for layer in &self.layers {
            t = layer.forward(&t);
        }
        t
    }
}

/// Frequency encoder layer: Conv2d([8,1]) → GELU → DConv (freq flattened) → 1x1 → GLU.
pub struct HEncLayer {
    pub conv_weight: Tensor,
    pub conv_bias: Option<Tensor>,
    pub dconv: DConv,
    pub rewrite_weight: Tensor,
    pub rewrite_bias: Option<Tensor>,
}

impl HEncLayer {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [B, C, Fr, T]
        let t = x.conv2d(
            &self.conv_weight,
            self.conv_bias.as_ref(),
            [STRIDE as i64, 1],
            [KERNEL_SIZE as i64 / 4, 0],
            [1, 1],
            1,
        );
        let t = gelu(&t);
        let dims = t.size();
        let (b, c, fr, tt) = (dims[0], dims[1], dims[2], dims[3]);
        // flatten freq into batch: [B, C, Fr, T] -> [B*Fr, C, T]
        let t = t.permute([0, 2, 1, 3]).reshape([b * fr, c, tt]);
        let t = self.dconv.forward(&t);
        let t = t.reshape([b, fr, c, tt]).permute([0, 2, 1, 3]);
        let t = t.conv2d(
            &self.rewrite_weight,
            self.rewrite_bias.as_ref(),
            [1, 1],
            [0, 0],
            [1, 1],
            1,
        );
        glu(&t, 1)
    }
}

/// Time encoder layer: right-pad to stride multiple → Conv1d(8) → GELU → DConv → 1x1 → GLU.
pub struct TEncLayer {
    pub conv_weight: Tensor,
    pub conv_bias: Option<Tensor>,
    pub dconv: DConv,
    pub rewrite_weight: Tensor,
    pub rewrite_bias: Option<Tensor>,
}

impl TEncLayer {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let le = x.size()[2];
        let x = if le % STRIDE as i64 != 0 {
            let pad_right = STRIDE as i64 - (le % STRIDE as i64);
            let zeros = Tensor::zeros(
                [x.size()[0], x.size()[1], pad_right],
                (Kind::Float, x.device()),
            );
            Tensor::cat(&[x.shallow_clone(), zeros], 2)
        } else {
            x.shallow_clone()
        };
        let t = x.conv1d(
            &self.conv_weight,
            self.conv_bias.as_ref(),
            [STRIDE as i64],
            [KERNEL_SIZE as i64 / 4],
            [1],
            1,
        );
        let t = gelu(&t);
        let t = self.dconv.forward(&t);
        let t = t.conv1d(
            &self.rewrite_weight,
            self.rewrite_bias.as_ref(),
            [1],
            [0],
            [1],
            1,
        );
        glu(&t, 1)
    }
}

/// Frequency decoder layer: skip add → 3x3 → GLU → DConv → ConvTranspose2d → trim → GELU.
pub struct HDecLayer {
    pub rewrite_weight: Tensor,
    pub rewrite_bias: Option<Tensor>,
    pub dconv: DConv,
    pub conv_tr_weight: Tensor,
    pub conv_tr_bias: Option<Tensor>,
    pub last: bool,
}

impl HDecLayer {
    pub fn forward(&self, x: &Tensor, skip: &Tensor, freq_target: i64) -> Tensor {
        let t = x + skip;
        let t = t.conv2d(
            &self.rewrite_weight,
            self.rewrite_bias.as_ref(),
            [1, 1],
            [1, 1],
            [1, 1],
            1,
        );
        let t = glu(&t, 1);
        let dims = t.size();
        let (b, c, fr, tt) = (dims[0], dims[1], dims[2], dims[3]);
        let t = t.permute([0, 2, 1, 3]).reshape([b * fr, c, tt]);
        let t = self.dconv.forward(&t);
        let t = t.reshape([b, fr, c, tt]).permute([0, 2, 1, 3]);
        let t = t.conv_transpose2d(
            &self.conv_tr_weight,
            self.conv_tr_bias.as_ref(),
            [STRIDE as i64, 1],
            [KERNEL_SIZE as i64 / 4, 0],
            [0, 0],
            1,
            [1, 1],
        );
        let t = if t.size()[2] > freq_target {
            t.narrow(2, 0, freq_target)
        } else {
            t
        };
        if self.last {
            t
        } else {
            gelu(&t)
        }
    }
}

/// Time decoder layer: trim skip → add → 3 → GLU → DConv → ConvTranspose1d → trim → GELU.
pub struct TDecLayer {
    pub rewrite_weight: Tensor,
    pub rewrite_bias: Option<Tensor>,
    pub dconv: DConv,
    pub conv_tr_weight: Tensor,
    pub conv_tr_bias: Option<Tensor>,
    pub last: bool,
}

impl TDecLayer {
    pub fn forward(&self, x: &Tensor, skip: &Tensor, time_target: i64) -> Tensor {
        let skip = if skip.size()[2] > x.size()[2] {
            skip.narrow(2, 0, x.size()[2])
        } else {
            skip.shallow_clone()
        };
        let t = x + skip;
        let t = t.conv1d(
            &self.rewrite_weight,
            self.rewrite_bias.as_ref(),
            [1],
            [1],
            [1],
            1,
        );
        let t = glu(&t, 1);
        let t = self.dconv.forward(&t);
        let t = t.conv_transpose1d(
            &self.conv_tr_weight,
            self.conv_tr_bias.as_ref(),
            [STRIDE as i64],
            [KERNEL_SIZE as i64 / 4],
            [0],
            1,
            [1],
        );
        let t = if t.size()[2] > time_target {
            t.narrow(2, 0, time_target)
        } else {
            t
        };
        if self.last {
            t
        } else {
            gelu(&t)
        }
    }
}

/// Number of channels after DConv compression (production uses ch / 8).
pub fn dconv_compress_ch(ch: usize) -> usize {
    ch / DCONV_COMP
}

/// Number of DConv residual layers (dilations 1, 2).
pub const DCONV_LAYERS: usize = DCONV_DEPTH;
