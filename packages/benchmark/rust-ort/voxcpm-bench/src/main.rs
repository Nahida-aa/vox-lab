use ndarray::{Array0, Array1, Array2, Array3, Array4, Array5, ArrayD};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::time::Instant;
use serde::Serialize;

const MODEL_DIR: &str =
    "/home/aa/repos/learn_ls/YouDub-webui/data/models/OpenBMB__VoxCPM2";
const FEAT_DIM: usize = 64;
const PATCH_SIZE: usize = 4;
const REF_PATCHES: usize = 25; // from a 3s reference audio

const TOKENS_SHORT: &[i64] = &[1, 21045, 59342, 1980, 1502, 1449, 74];
const TOKENS_MEDIUM: &[i64] = &[1, 13393, 1410, 1348, 5919, 2676, 72, 2965, 59361, 59328, 1717, 1421, 1348, 5694, 1377, 1358, 7138, 1384, 4740, 1358, 51487, 3985, 72];
const TOKENS_LONG: &[i64] = &[1, 46715, 17036, 1410, 33342, 1358, 2369, 1554, 4697, 1384, 1758, 72, 6051, 5069, 4446, 7639, 1385, 6035, 11077, 1384, 30991, 9867, 59342, 17251, 10301, 1580, 2563, 22421, 6395, 1377, 4965, 2243, 72, 3348, 19963, 1502, 7247, 1715, 8592, 1384, 9006, 4073, 2291, 4936, 72];

#[derive(Serialize)]
struct BenchmarkResult {
    engine: String,
    device: String,
    text_key: String,
    text_len: usize,
    load_time_s: f64,
    generate_time_s: f64,
    total_time_s: f64,
    output_samples: usize,
    output_duration_s: f64,
    auto_patches: usize,
    rtf: f64,
}

fn load_model(path: &str) -> ort::Result<Session> {
    Session::builder()?.commit_from_file(&Path::new(MODEL_DIR).join(path))
}

fn randn(n: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n).map(|_| {
        let u: f32 = rng.r#gen();
        let v: f32 = rng.r#gen();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }).collect()
}

fn main() -> ort::Result<()> {
    println!("=== Rust ort full VoxCPM pipeline benchmark ===\n");

    let t0 = Instant::now();
    println!("Loading VAE Encoder...");
    let mut _vae_enc = load_model("audio_vae_encoder.onnx")?;
    println!("  {:.1}s", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    println!("Loading Prefill (2B)...");
    let mut prefill = load_model("voxcpm2_prefill.onnx")?;
    println!("  {:.1}s", t1.elapsed().as_secs_f64());

    let t2 = Instant::now();
    println!("Loading Decode Step...");
    let mut decode = load_model("voxcpm2_decode_step.onnx")?;
    println!("  {:.2}s", t2.elapsed().as_secs_f64());

    let t3 = Instant::now();
    println!("Loading VAE Decoder...");
    let mut vae_dec = load_model("audio_vae_decoder.onnx")?;
    println!("  {:.2}s", t3.elapsed().as_secs_f64());

    let load_time = t0.elapsed().as_secs_f64();
    println!("Total load: {:.1}s\n", load_time);

    // Skip VAE encode: use dummy ref patches
    let ref_patches: Vec<Vec<f32>> = (0..REF_PATCHES)
        .map(|_| randn(FEAT_DIM))
        .collect();

    let cases: [(&str, &[i64], usize); 2] = [
        ("short", TOKENS_SHORT, std::cmp::max(20, (TOKENS_SHORT.len() as f64 * 6.0).ceil() as usize)),
        ("medium", TOKENS_MEDIUM, std::cmp::max(20, (TOKENS_MEDIUM.len() as f64 * 6.0).ceil() as usize)),
    ];

    let mut results: Vec<BenchmarkResult> = Vec::new();

    for &(text_key, text_ids, auto_patches) in &cases {
        let text_len = text_ids.len();
        let seq_len = 1 + REF_PATCHES + 1 + text_len + 1;

        println!("--- {} | {} tokens, seq_len={}, auto_patches={} ---", text_key, text_len, seq_len, auto_patches);

        let t_gen = Instant::now();

        // Build inputs
        let mut text_tokens: Vec<i64> = Vec::with_capacity(seq_len);
        let mut text_mask: Vec<i32> = Vec::with_capacity(seq_len);
        let mut feat_mask: Vec<i32> = Vec::with_capacity(seq_len);
        let mut flat_feat = vec![0.0f32; seq_len * PATCH_SIZE * FEAT_DIM];

        text_tokens.push(103); text_mask.push(1); feat_mask.push(0);

        for patch in &ref_patches {
            let pos = text_tokens.len();
            text_tokens.push(0); text_mask.push(0); feat_mask.push(1);
            for p in 0..PATCH_SIZE {
                for d in 0..FEAT_DIM {
                    flat_feat[pos * PATCH_SIZE * FEAT_DIM + p * FEAT_DIM + d] = patch[d];
                }
            }
        }

        text_tokens.push(104); text_mask.push(1); feat_mask.push(0);

        for &id in text_ids {
            text_tokens.push(id); text_mask.push(1); feat_mask.push(0);
        }

        text_tokens.push(101); text_mask.push(1); feat_mask.push(0);

        // Prefill - debug shapes
        let a2 = Array2::from_shape_vec((1, seq_len), text_tokens.clone());
        println!("  text {}x{} -> {:?}", 1, seq_len, a2.as_ref().map(|_| "ok"));
        let a2m = Array2::from_shape_vec((1, seq_len), text_mask.clone());
        let a2f = Array2::from_shape_vec((1, seq_len), feat_mask.clone());
        let a4 = Array4::from_shape_vec((1, seq_len, PATCH_SIZE, FEAT_DIM), flat_feat.clone());
        println!("  text_mask {:?} feat_mask {:?} feat {:?}",
            a2m.as_ref().map(|_| "ok"), a2f.as_ref().map(|_| "ok"),
            a4.as_ref().map(|_| "ok"));

        let pf_out = prefill.run(ort::inputs![
            "text" => Tensor::from_array(a2.unwrap().into_dyn()).unwrap(),
            "text_mask" => Tensor::from_array(a2m.unwrap().into_dyn()).unwrap(),
            "feat" => Tensor::from_array(a4.unwrap().into_dyn()).unwrap(),
            "feat_mask" => Tensor::from_array(a2f.unwrap().into_dyn()).unwrap(),
        ])?;

        let dit_hidden_data: Vec<f32> = pf_out["dit_hidden"].try_extract_tensor::<f32>()?.1.to_vec();
        let base_k_data: Vec<f32> = pf_out["base_next_keys"].try_extract_tensor::<f32>()?.1.to_vec();
        let base_v_data: Vec<f32> = pf_out["base_next_values"].try_extract_tensor::<f32>()?.1.to_vec();
        let res_k_data: Vec<f32> = pf_out["residual_next_keys"].try_extract_tensor::<f32>()?.1.to_vec();
        let res_v_data: Vec<f32> = pf_out["residual_next_values"].try_extract_tensor::<f32>()?.1.to_vec();
        let prefix_data: Vec<f32> = pf_out["prefix_feat_cond"].try_extract_tensor::<f32>()?.1.to_vec();

        let mut cache_len = seq_len;
        let mut k_sz = [1, 28, 2, cache_len, 128];
        let mut k_sz_res = [1, 8, 2, cache_len, 128];

        let mut dit_hidden = dit_hidden_data;
        let mut base_keys = base_k_data;
        let mut base_vals = base_v_data;
        let mut res_keys = res_k_data;
        let mut res_vals = res_v_data;
        let mut prefix = prefix_data;

        let mut pred_patches: Vec<Vec<f32>> = Vec::new();

        for step in 0..auto_patches {
            k_sz[3] = cache_len;
            k_sz_res[3] = cache_len;

            let noise = randn(PATCH_SIZE * FEAT_DIM);

            let dec_out = decode.run(ort::inputs![
                "dit_hidden" => Tensor::from_array(Array2::from_shape_vec((1, 2048), dit_hidden.clone()).unwrap().into_dyn()).unwrap(),
                "base_next_keys" => Tensor::from_array(Array5::from_shape_vec(k_sz, base_keys.clone()).unwrap().into_dyn()).unwrap(),
                "base_next_values" => Tensor::from_array(Array5::from_shape_vec(k_sz, base_vals.clone()).unwrap().into_dyn()).unwrap(),
                "residual_next_keys" => Tensor::from_array(Array5::from_shape_vec(k_sz_res, res_keys.clone()).unwrap().into_dyn()).unwrap(),
                "residual_next_values" => Tensor::from_array(Array5::from_shape_vec(k_sz_res, res_vals.clone()).unwrap().into_dyn()).unwrap(),
                "prefix_feat_cond" => Tensor::from_array(Array3::from_shape_vec((1, 4, FEAT_DIM), prefix.clone()).unwrap().into_dyn()).unwrap(),
                "noise" => Tensor::from_array(Array3::from_shape_vec((1, PATCH_SIZE, FEAT_DIM), noise).unwrap().into_dyn()).unwrap(),
                "cfg_value" => Tensor::from_array(Array0::from_elem((), 2.0f32).into_dyn()).unwrap(),
            ])?;

            let pred_feat: Vec<f32> = dec_out["pred_feat"].try_extract_tensor::<f32>()?.1.to_vec();
            dit_hidden = dec_out["new_dit_hidden"].try_extract_tensor::<f32>()?.1.to_vec();
            base_keys = dec_out["new_base_next_keys"].try_extract_tensor::<f32>()?.1.to_vec();
            base_vals = dec_out["new_base_next_values"].try_extract_tensor::<f32>()?.1.to_vec();
            res_keys = dec_out["new_residual_next_keys"].try_extract_tensor::<f32>()?.1.to_vec();
            res_vals = dec_out["new_residual_next_values"].try_extract_tensor::<f32>()?.1.to_vec();

            pred_patches.push(pred_feat.clone());
            prefix = pred_feat;
            cache_len += 1;

            if (step + 1) % 25 == 0 || step == auto_patches - 1 {
                println!("  step {}/{}", step + 1, auto_patches);
            }
        }

        // VAE Decoder
        let num_patches = pred_patches.len();
        let z_len = num_patches * PATCH_SIZE;
        let mut z_data = vec![0.0f32; FEAT_DIM * z_len];

        for t in 0..num_patches {
            for p in 0..PATCH_SIZE {
                for d in 0..FEAT_DIM {
                    z_data[d * z_len + t * PATCH_SIZE + p] = pred_patches[t][p * FEAT_DIM + d];
                }
            }
        }

        let ae_out = vae_dec.run(ort::inputs![
            "z" => Tensor::from_array(Array3::from_shape_vec((1, FEAT_DIM, z_len), z_data).unwrap().into_dyn()).unwrap(),
        ])?;

        let output_samples = ae_out["audio"].try_extract_tensor::<f32>()?.1.len();

        let gen_time = t_gen.elapsed().as_secs_f64();
        let out_dur = output_samples as f64 / 48000.0;
        let rtf = gen_time / out_dur;

        println!("  Done: {:.3}s output {:.2}s RTF {:.3}", gen_time, out_dur, rtf);

        let r = BenchmarkResult {
            engine: "rust".into(),
            device: "cpu".into(),
            text_key: text_key.into(),
            text_len,
            load_time_s: (load_time * 1000.0).round() / 1000.0,
            generate_time_s: (gen_time * 1000.0).round() / 1000.0,
            total_time_s: ((load_time + gen_time) * 1000.0).round() / 1000.0,
            output_samples,
            output_duration_s: (out_dur * 1000.0).round() / 1000.0,
            auto_patches: num_patches,
            rtf: (rtf * 1000.0).round() / 1000.0,
        };
        results.push(r);

        // Save after each case so partial results survive timeout
        let out = Path::new(MODEL_DIR)
            .parent().unwrap().parent().unwrap().parent().unwrap()
            .join("packages").join("benchmark").join("VC").join("VoxCPM2").join("results")
            .join("rs-onnx-cpu.json");
        let partial = serde_json::to_string_pretty(&results).unwrap();
        std::fs::write(&out, &partial).ok();
    }

    println!("\nAll done.");
    Ok(())
}
