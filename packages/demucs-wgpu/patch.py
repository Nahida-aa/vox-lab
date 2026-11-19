from pathlib import Path

path = Path(__file__).with_name("src") / "lib.rs"
src = path.read_text()

old_header = """//! 这样不改上游代码也能读我们已有的权重文件。

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use demucs_core::demucs::pipeline::SeparateOptions;
use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection};
"""

new_header = """//! 这样不改上游代码也能读我们已有的权重文件。
//!
//! 推理走上游的 streaming 路径（`separate_streaming_file` + `FloatWavWriter`）：
//! 输入按窗口读、stem 帧从 OLA flush 直接写盘，不再整轨驻留内存，峰值为
//! O(segment) 而不是 O(音轨长度)。多轮基准只有最后一轮落盘。

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use demucs_core::audio::{probe_audio, FloatWavWriter};
use demucs_core::demucs::pipeline::SeparateOptions;
use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection, AUDIO_CHANNELS};
"""

old_run = """    let input = cli.input.as_ref().context("Input file required")?;
    let output = cli.output.as_ref().context("Output directory required")?;
    std::fs::create_dir_all(output)?;

    let (left, right, sample_rate) = read_wav(input)?;
    eprintln!("Reading {} ({} Hz)", input.display(), sample_rate);

    // Deterministic by default: one fixed pass, no random shift averaging.
    let options = SeparateOptions {
        shifts: 0,
        ..Default::default()
    };

    let mut last_stems = None;
    for round in 1..=cli.benchmark_rounds {
        eprintln!("--- Round {}/{} ---", round, cli.benchmark_rounds);
        let t0 = Instant::now();
        let stems = demucs
            .separate_with_options(&left, &right, sample_rate, &options, &mut |p| {
                println!("({}%)", p.percent());
            })
            .map_err(|e| anyhow::anyhow!("separation failed: {e}"))?;
        let gen_secs = t0.elapsed().as_secs_f64();
        println!("Benchmark-Gen-Time-Round{}: {:.3}", round, gen_secs);
        if round == cli.benchmark_rounds {
            println!("Benchmark-Gen-Time: {:.3}", gen_secs);
            last_stems = Some(stems);
        }
    }

    let stems = last_stems.context("no inference round produced stems")?;
    for (i, stem) in stems.iter().enumerate() {
        let path = output.join(format!("target_{}_{}.wav", i, stem.id.as_str()));
        write_wav(&path, &stem.left, &stem.right, sample_rate)?;
        eprintln!("Wrote {}", path.display());
    }
    eprintln!("Done!");
    Ok(())
}
"""

new_run = """    let input = cli.input.as_ref().context("Input file required")?;
    let output = cli.output.as_ref().context("Output directory required")?;
    std::fs::create_dir_all(output)?;

    let info = probe_audio(input)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", input.display()))?;
    eprintln!("Reading {} ({} Hz)", input.display(), info.sample_rate);
    let sample_rate = demucs.sample_rate() as u32;
    let sources: Vec<String> = demucs.sources().to_vec();

    // Deterministic by default: one fixed pass, no random shift averaging.
    let options = SeparateOptions {
        shifts: 0,
        ..Default::default()
    };

    for round in 1..=cli.benchmark_rounds {
        eprintln!("--- Round {}/{} ---", round, cli.benchmark_rounds);
        let final_round = round == cli.benchmark_rounds;

        // Earlier rounds are pure timing passes, so their frames are dropped and
        // only the last round writes WAVs.
        let mut writers: Vec<Option<FloatWavWriter>> = sources
            .iter()
            .enumerate()
            .map(|(index, name)| {
                if !final_round {
                    return Ok(None);
                }
                let path = output.join(format!("target_{}_{}.wav", index, name));
                FloatWavWriter::create(&path, sample_rate, 2)
                    .map(Some)
                    .map_err(|e| anyhow::anyhow!("creating {}: {e}", path.display()))
            })
            .collect::<Result<Vec<_>>>()?;

        let t0 = Instant::now();
        demucs
            .separate_streaming_file(
                input,
                &options,
                &mut |p| println!("({}%)", p.percent()),
                &mut |_t, frame| {
                    for (index, slot) in writers.iter_mut().enumerate() {
                        if let Some(writer) = slot {
                            let base = index * AUDIO_CHANNELS;
                            writer.write_frame(&[frame[base], frame[base + 1]])?;
                        }
                    }
                    Ok(())
                },
            )
            .map_err(|e| anyhow::anyhow!("separation failed: {e}"))?;
        let gen_secs = t0.elapsed().as_secs_f64();

        for (index, slot) in writers.iter_mut().enumerate() {
            if let Some(writer) = slot.take() {
                let path = output.join(format!("target_{}_{}.wav", index, sources[index]));
                writer
                    .finish()
                    .map_err(|e| anyhow::anyhow!("writing {}: {e}", path.display()))?;
                eprintln!("Wrote {}", path.display());
            }
        }

        println!("Benchmark-Gen-Time-Round{}: {:.3}", round, gen_secs);
        if final_round {
            println!("Benchmark-Gen-Time: {:.3}", gen_secs);
        }
    }

    eprintln!("Done!");
    Ok(())
}
"""

for name, old, new in (("header", old_header, new_header), ("run", old_run, new_run)):
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"{name}: expected 1 occurrence, found {count}")
    src = src.replace(old, new)

path.write_text(src)
print(f"patched, {len(src.splitlines())} lines")
