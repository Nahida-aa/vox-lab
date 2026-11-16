"""cli — VoxCPM 实验入口。

  python -m voxcpm_pytorch.cli generate --text "..." --out out.wav
  python -m voxcpm_pytorch.cli profile  --text "..." 
  python -m voxcpm_pytorch.cli dtype    --text "..." --dtypes bf16 fp32

设计目标：给出「值不值得做自研 GPU kernel」所需的两组数据 —— 分段耗时与精度代价。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import core


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-dir", default=str(core.DEFAULT_MODEL_DIR), help="VoxCPM 模型目录")
    p.add_argument("--device", default="cpu", help="cpu / cuda / auto")
    p.add_argument("--text", default="这是一段用于测试的中文语音。", help="待合成文本")
    p.add_argument("--dit-steps", type=int, default=10, help="CFM 迭代步数")
    p.add_argument("--cfg", type=float, default=2.0, help="cfg_value")
    p.add_argument("--seed", type=int, default=42, help="随机种子（None 表示不固定）")
    p.add_argument("--max-len", type=int, default=120, help="自回归 token 上界（官方默认 4096，实验必须限流）")


def cmd_generate(args: argparse.Namespace) -> None:
    model = core.load_model(args.model_dir, device=args.device, optimize=args.optimize)
    wav, times = core.timed_generate(
        model, args.text, rounds=args.rounds,
        cfg_value=args.cfg, dit_steps=args.dit_steps, seed=args.seed, max_len=args.max_len,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    core.save_wav(out, wav)
    audio_s = len(wav) / core.DEFAULT_SAMPLE_RATE
    best = min(times)
    print(f"load   {getattr(model, '_vox_lab_load_s', 0.0):.2f}s")
    for i, t in enumerate(times, 1):
        print(f"round{i} {t:.2f}s  RTF {t / audio_s:.2f}")
    print(f"best   {best:.2f}s  RTF {best / audio_s:.2f}  ({audio_s:.2f}s audio)")
    print(f"wrote  {out}")


def cmd_profile(args: argparse.Namespace) -> None:
    model = core.load_model(
        args.model_dir, device=args.device, optimize=False, dtype=args.dtype
    )
    times = core.stage_breakdown(
        model, args.text, cfg_value=args.cfg, dit_steps=args.dit_steps,
        seed=args.seed, max_len=args.max_len,
    )
    print(f"{'stage':12} {'ms':>10} {'share':>8}")
    for name, ms, share in times.rows():
        print(f"{name:12} {ms:10.1f} {share:7.1f}%")
    print(f"{'total':12} {times.total_ms:10.1f}  (profiler self-time sum)")
    print(f"wall {times.wall_s:.2f}s  audio {times.audio_s:.2f}s  RTF {times.rtf:.2f}")


def cmd_dtype(args: argparse.Namespace) -> None:
    results = core.dtype_experiment(
        model_dir=args.model_dir,
        text=args.text,
        device=args.device,
        dtypes=tuple(args.dtypes),
        dit_steps=args.dit_steps,
        max_len=args.max_len,
    )
    print(f"{'dtype':8} {'load_s':>8} {'gen_s':>8} {'audio_s':>8} {'SNR(dB)':>9}")
    for r in results:
        audio_s = r.samples / core.DEFAULT_SAMPLE_RATE
        snr = "reference" if r.snr_db == float("inf") else f"{r.snr_db:.1f}"
        print(f"{r.dtype:8} {r.load_s:8.2f} {r.gen_s:8.2f} {audio_s:8.2f} {snr:>9}")
    if args.json:
        print(json.dumps([r.__dict__ for r in results], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM PyTorch 实验工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="合成一段文本并写 wav")
    _add_common(g)
    g.add_argument("--out", default="tmp/voxcpm-pytorch/out.wav")
    g.add_argument("--rounds", type=int, default=1)
    g.add_argument("--optimize", action="store_true", help="启用 torch.compile（会干扰 profiling）")
    g.set_defaults(func=cmd_generate)

    pr = sub.add_parser("profile", help="按阶段聚合耗时（LM / residual LM / CFM / VAE）")
    _add_common(pr)
    pr.add_argument("--dtype", default=None, help="覆盖权重精度：fp32 / bf16 / fp16（默认用 checkpoint 的 bf16）")
    pr.set_defaults(func=cmd_profile)

    d = sub.add_parser("dtype", help="对比 bf16 / fp16 / fp32 的精度与耗时")
    _add_common(d)
    d.add_argument("--dtypes", nargs="+", default=["bf16", "fp32"])
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_dtype)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
