//! Safetensors → tch::Tensor weight loader.
//!
//! Ported from `demucs-core/src/weights/load.rs` (Burn), but using PyTorch-native
//! layouts: NO Linear transpose, NO MHA `in_proj` split. tch-rs wraps libtorch,
//! so conv/linear/norm weights are used exactly as stored in the file.

use tch::{Device, Kind, Tensor};

use crate::model::conv::{
    DConv, DConvLayer, HDecLayer, HEncLayer, TDecLayer, TEncLayer, DCONV_LAYERS,
};
use crate::model::htdemucs::{HTDemucs, HyperParameters};
use crate::model::metadata::ModelInfo;
use crate::model::transformer::{
    create_layer_scale, Conv1d as ProjConv1d, CrossAttentionLayer, CrossDomainTransformer,
    GroupNorm, LayerNorm, Linear, MultiHeadAttention, SelfAttentionLayer, TransformerLayer,
    N_HEADS,
};
use crate::weights::tensor_store::TensorStore;
use crate::weights::WeightError;
use crate::T_LAYERS;

fn st_to_tensor(t: crate::weights::tensor_store::StoredTensor, device: Device) -> Tensor {
    let shape: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
    Tensor::from_slice(&t.data)
        .reshape(&shape)
        .to_kind(Kind::Float)
        .to_device(device)
}

fn take_tensor(store: &mut TensorStore, key: &str) -> Result<Tensor, WeightError> {
    let device = store.device();
    Ok(st_to_tensor(store.take(key)?, device))
}

fn maybe_take_tensor(store: &mut TensorStore, key: &str) -> Option<Tensor> {
    let device = store.device();
    store.take(key).ok().map(|t| st_to_tensor(t, device))
}

// ─── Primitive loaders (PyTorch layout, no transformation) ─────────────────

fn load_conv(
    store: &mut TensorStore,
    prefix: &str,
) -> Result<(Tensor, Option<Tensor>), WeightError> {
    let w = take_tensor(store, &format!("{}.weight", prefix))?;
    let b = maybe_take_tensor(store, &format!("{}.bias", prefix));
    Ok((w, b))
}

fn load_norm(
    store: &mut TensorStore,
    prefix: &str,
) -> Result<(Option<Tensor>, Option<Tensor>), WeightError> {
    let w = maybe_take_tensor(store, &format!("{}.weight", prefix));
    let b = maybe_take_tensor(store, &format!("{}.bias", prefix));
    Ok((w, b))
}

fn load_layer_scale(store: &mut TensorStore, prefix: &str) -> Result<Tensor, WeightError> {
    take_tensor(store, &format!("{}.scale", prefix))
}

fn load_linear(store: &mut TensorStore, prefix: &str) -> Result<Linear, WeightError> {
    let w = take_tensor(store, &format!("{}.weight", prefix))?;
    let b = maybe_take_tensor(store, &format!("{}.bias", prefix));
    Ok(Linear { weight: w, bias: b })
}

fn load_layernorm(store: &mut TensorStore, prefix: &str, d_model: i64) -> Result<LayerNorm, WeightError> {
    let (weight, bias) = load_norm(store, prefix)?;
    Ok(LayerNorm { weight, bias, d_model })
}

fn load_groupnorm(store: &mut TensorStore, prefix: &str) -> Result<GroupNorm, WeightError> {
    let (weight, bias) = load_norm(store, prefix)?;
    Ok(GroupNorm { weight, bias })
}

// ─── Composite loaders ─────────────────────────────────────────────────────

fn load_dconv_layer(
    store: &mut TensorStore,
    prefix: &str,
    dilation: i64,
) -> Result<DConvLayer, WeightError> {
    // PyTorch sequential: .0 conv1, .1 norm1, .3 conv2, .4 norm2, .6 scale
    let (conv1_weight, conv1_bias) = load_conv(store, &format!("{}.0", prefix))?;
    let (norm1_weight, norm1_bias) = load_norm(store, &format!("{}.1", prefix))?;
    let (conv2_weight, conv2_bias) = load_conv(store, &format!("{}.3", prefix))?;
    let (norm2_weight, norm2_bias) = load_norm(store, &format!("{}.4", prefix))?;
    let scale = take_tensor(store, &format!("{}.6.scale", prefix))?;
    // conv1 is a depthwise/grouped conv when channels are compressed by ch/8.
    // groups is derived from weight shape [out, in/groups, k] at forward time,
    // but Burn used groups=1 with ch/8 output channels. Keep groups=1.
    Ok(DConvLayer {
        conv1_weight,
        conv1_bias,
        norm1_weight,
        norm1_bias,
        conv2_weight,
        conv2_bias,
        norm2_weight,
        norm2_bias,
        scale,
        dilation,
        groups: 1,
    })
}

fn load_dconv(store: &mut TensorStore, prefix: &str) -> Result<DConv, WeightError> {
    let mut layers = Vec::new();
    for j in 0..DCONV_LAYERS {
        layers.push(load_dconv_layer(
            store,
            &format!("{}.{}", prefix, j),
            1i64 << j,
        )?);
    }
    Ok(DConv { layers })
}

fn load_henc(store: &mut TensorStore, prefix: &str) -> Result<HEncLayer, WeightError> {
    let (conv_weight, conv_bias) = load_conv(store, &format!("{}.conv", prefix))?;
    let dconv = load_dconv(store, &format!("{}.dconv.layers", prefix))?;
    let (rewrite_weight, rewrite_bias) = load_conv(store, &format!("{}.rewrite", prefix))?;
    Ok(HEncLayer {
        conv_weight,
        conv_bias,
        dconv,
        rewrite_weight,
        rewrite_bias,
    })
}

fn load_tenc(store: &mut TensorStore, prefix: &str) -> Result<TEncLayer, WeightError> {
    let (conv_weight, conv_bias) = load_conv(store, &format!("{}.conv", prefix))?;
    let dconv = load_dconv(store, &format!("{}.dconv.layers", prefix))?;
    let (rewrite_weight, rewrite_bias) = load_conv(store, &format!("{}.rewrite", prefix))?;
    Ok(TEncLayer {
        conv_weight,
        conv_bias,
        dconv,
        rewrite_weight,
        rewrite_bias,
    })
}

fn load_hdec(store: &mut TensorStore, prefix: &str, last: bool) -> Result<HDecLayer, WeightError> {
    let (rewrite_weight, rewrite_bias) = load_conv(store, &format!("{}.rewrite", prefix))?;
    let dconv = load_dconv(store, &format!("{}.dconv.layers", prefix))?;
    let (conv_tr_weight, conv_tr_bias) = load_conv(store, &format!("{}.conv_tr", prefix))?;
    Ok(HDecLayer {
        rewrite_weight,
        rewrite_bias,
        dconv,
        conv_tr_weight,
        conv_tr_bias,
        last,
    })
}

fn load_tdec(store: &mut TensorStore, prefix: &str, last: bool) -> Result<TDecLayer, WeightError> {
    let (rewrite_weight, rewrite_bias) = load_conv(store, &format!("{}.rewrite", prefix))?;
    let dconv = load_dconv(store, &format!("{}.dconv.layers", prefix))?;
    let (conv_tr_weight, conv_tr_bias) = load_conv(store, &format!("{}.conv_tr", prefix))?;
    Ok(TDecLayer {
        rewrite_weight,
        rewrite_bias,
        dconv,
        conv_tr_weight,
        conv_tr_bias,
        last,
    })
}

// ─── Attention / transformer loaders ───────────────────────────────────────

fn load_mha(store: &mut TensorStore, prefix: &str, d_model: i64) -> Result<MultiHeadAttention, WeightError> {
    let in_proj_weight = take_tensor(store, &format!("{}.in_proj_weight", prefix))?;
    let in_proj_bias = maybe_take_tensor(store, &format!("{}.in_proj_bias", prefix));
    let out_proj = load_linear(store, &format!("{}.out_proj", prefix))?;
    Ok(MultiHeadAttention {
        in_proj_weight,
        in_proj_bias,
        out_proj,
        n_heads: N_HEADS,
        d_model,
    })
}

fn load_self_attn_layer(
    store: &mut TensorStore,
    prefix: &str,
    d_model: i64,
) -> Result<SelfAttentionLayer, WeightError> {
    let norm1 = load_layernorm(store, &format!("{}.norm1", prefix), d_model)?;
    let attn = load_mha(store, &format!("{}.self_attn", prefix), d_model)?;
    let gamma_1 = create_layer_scale(load_layer_scale(store, &format!("{}.gamma_1", prefix))?);
    let norm2 = load_layernorm(store, &format!("{}.norm2", prefix), d_model)?;
    let linear1 = load_linear(store, &format!("{}.linear1", prefix))?;
    let linear2 = load_linear(store, &format!("{}.linear2", prefix))?;
    let gamma_2 = create_layer_scale(load_layer_scale(store, &format!("{}.gamma_2", prefix))?);
    let norm_out = load_optional_groupnorm(store, prefix)?;
    Ok(SelfAttentionLayer {
        norm1,
        attn,
        gamma_1,
        norm2,
        linear1,
        linear2,
        gamma_2,
        norm_out,
    })
}

fn load_cross_attn_layer(
    store: &mut TensorStore,
    prefix: &str,
    d_model: i64,
) -> Result<CrossAttentionLayer, WeightError> {
    let norm1 = load_layernorm(store, &format!("{}.norm1", prefix), d_model)?;
    let norm2 = load_layernorm(store, &format!("{}.norm2", prefix), d_model)?;
    let attn = load_mha(store, &format!("{}.cross_attn", prefix), d_model)?;
    let gamma_1 = create_layer_scale(load_layer_scale(store, &format!("{}.gamma_1", prefix))?);
    let norm3 = load_layernorm(store, &format!("{}.norm3", prefix), d_model)?;
    let linear1 = load_linear(store, &format!("{}.linear1", prefix))?;
    let linear2 = load_linear(store, &format!("{}.linear2", prefix))?;
    let gamma_2 = create_layer_scale(load_layer_scale(store, &format!("{}.gamma_2", prefix))?);
    let norm_out = load_optional_groupnorm(store, prefix)?;
    Ok(CrossAttentionLayer {
        norm1,
        norm2,
        attn,
        gamma_1,
        norm3,
        linear1,
        linear2,
        gamma_2,
        norm_out,
    })
}

/// `norm_out` keys may exist in the file even if the layer has no norm_out.
fn load_optional_groupnorm(
    store: &mut TensorStore,
    prefix: &str,
) -> Result<Option<GroupNorm>, WeightError> {
    let key = format!("{}.norm_out.weight", prefix);
    if store.has_key(&key) {
        Ok(Some(load_groupnorm(store, &format!("{}.norm_out", prefix))?))
    } else {
        store.skip_prefix(&format!("{}.norm_out", prefix));
        Ok(None)
    }
}

fn load_transformer_layer(
    store: &mut TensorStore,
    prefix: &str,
    d_model: i64,
    is_self: bool,
) -> Result<TransformerLayer, WeightError> {
    if is_self {
        Ok(TransformerLayer::SelfAttn(load_self_attn_layer(
            store, prefix, d_model,
        )?))
    } else {
        Ok(TransformerLayer::CrossAttn(load_cross_attn_layer(
            store, prefix, d_model,
        )?))
    }
}

fn load_conv1d_opt(store: &mut TensorStore, prefix: &str) -> Result<Option<ProjConv1d>, WeightError> {
    let key = format!("{}.weight", prefix);
    if !store.has_key(&key) {
        return Ok(None);
    }
    let (weight, bias) = load_conv(store, prefix)?;
    Ok(Some(ProjConv1d { weight, bias }))
}

fn load_cross_domain_transformer(
    store: &mut TensorStore,
    d_model: i64,
) -> Result<CrossDomainTransformer, WeightError> {
    let channel_upsampler = load_conv1d_opt(store, "channel_upsampler")?;
    let channel_downsampler = load_conv1d_opt(store, "channel_downsampler")?;
    let channel_upsampler_t = load_conv1d_opt(store, "channel_upsampler_t")?;
    let channel_downsampler_t = load_conv1d_opt(store, "channel_downsampler_t")?;

    let norm_in = load_layernorm(store, "crosstransformer.norm_in", d_model)?;
    let norm_in_t = load_layernorm(store, "crosstransformer.norm_in_t", d_model)?;

    let mut layers = Vec::new();
    let mut layers_t = Vec::new();
    for i in 0..T_LAYERS {
        let is_self = i % 2 == 0;
        layers.push(load_transformer_layer(
            store,
            &format!("crosstransformer.layers.{}", i),
            d_model,
            is_self,
        )?);
        layers_t.push(load_transformer_layer(
            store,
            &format!("crosstransformer.layers_t.{}", i),
            d_model,
            is_self,
        )?);
    }

    Ok(CrossDomainTransformer {
        d_model,
        norm_in,
        norm_in_t,
        channel_upsampler,
        channel_downsampler,
        channel_upsampler_t,
        channel_downsampler_t,
        layers,
        layers_t,
    })
}

// ─── Top-level model loader ─────────────────────────────────────────────────

fn load_htdemucs_from_store(
    store: &mut TensorStore,
    hp: &HyperParameters,
) -> Result<HTDemucs, WeightError> {
    let mut encoders = Vec::new();
    for i in 0..crate::DEPTH {
        encoders.push(load_henc(store, &format!("encoder.{}", i))?);
    }
    let mut tencoders = Vec::new();
    for i in 0..crate::DEPTH {
        tencoders.push(load_tenc(store, &format!("tencoder.{}", i))?);
    }
    // PyTorch numbering: decoder.0 is the deepest (384→192), decoder.3 the
    // shallowest (48→output). Model.decoders[0] must receive decoder.0.
    // This matches demucs-core's loader (positive i over decoders.iter_mut()).
    let mut decoders = Vec::new();
    let mut tdecoders = Vec::new();
    for i in 0..crate::DEPTH {
        let last = i == crate::DEPTH - 1;
        decoders.push(load_hdec(store, &format!("decoder.{}", i), last)?);
        tdecoders.push(load_tdec(store, &format!("tdecoder.{}", i), last)?);
    }

    // freq_emb: ScaledEmbedding(scale=10) — bake ×10 into the weights on load.
    let mut emb = take_tensor(store, "freq_emb.embedding.weight")?;
    emb = emb * 10.0;

    let crosstransformer = load_cross_domain_transformer(store, hp.bottom_channels as i64)?;

    Ok(HTDemucs {
        encoders,
        tencoders,
        crosstransformer,
        decoders,
        tdecoders,
        freq_emb_weight: emb,
        n_sources: hp.n_sources,
    })
}

/// Load one or more HTDemucs models from safetensors bytes.
/// `htdemucs_ft` has 4 sub-models (one per stem) in a single file.
pub fn load_model(
    bytes: &[u8],
    info: &ModelInfo,
    device: Device,
) -> Result<Vec<HTDemucs>, WeightError> {
    let hp = HyperParameters::from_model_info(info);
    let mut models = Vec::new();

    for sig in info.signatures {
        let mut store = TensorStore::from_bytes(bytes, sig, device)?;
        let model = load_htdemucs_from_store(&mut store, &hp)?;
        let remaining = store.remaining_keys();
        if !remaining.is_empty() {
            return Err(WeightError::ShapeMismatch(format!(
                "unused keys for signature {}: {:?}",
                sig,
                remaining.iter().take(50).collect::<Vec<_>>()
            )));
        }
        models.push(model);
    }

    Ok(models)
}
