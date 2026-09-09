#!/usr/bin/env python3
"""Run subtitle-py.py OCR multiple times on same video frames.

Records:
  - per-run total inference time (avg + stddev)
  - per-frame inference time distribution
  - per-segment: text and confidence variance across runs

Usage:
  python benchmark-py-repeat.py --runs 5 --text-score 0.5 --subtitle-only
  python benchmark-py-repeat.py --runs 5 --frames-dir /path/to/frames/
"""
import argparse, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FRAMES = REPO_ROOT / "packages" / "benchmark" / "ocr" / "results" / "frames-pre-2fps"
OUT_DIR = REPO_ROOT / "packages" / "subtitle-ocr" / "docs"
SUBTITLE_PY = REPO_ROOT / "packages" / "subtitle-ocr" / "subtitle-py.py"

# Call subtitle-py.py via subprocess (one fresh process per frame) — this
# ensures no process-state leaks between frames/runs.
def call_ocr_frame(frame_path: str, text_score=None, subtitle_only=False):
    args = [sys.executable, str(SUBTITLE_PY), frame_path]
    if text_score is not None:
        args.extend(["--text-score", str(text_score)])
    if subtitle_only:
        args.append("--subtitle-only")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        return [], 0.0
    try:
        data = json.loads(r.stdout)
        return data.get("lines", []), float(data.get("inference_ms", 0.0))
    except Exception:
        return [], 0.0


def get_frame_paths(frames_dir: Path):
    return sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])


def do_runs(frames, runs, text_score, subtitle_only):
    per_pass = []
    for run in range(runs):
        per_frame_ms = []
        frame_lines = []
        t_start = time.perf_counter()
        for fp in frames:
            t0 = time.perf_counter()
            lines, _inference_ms_inner = call_ocr_frame(
                str(fp), text_score=text_score, subtitle_only=subtitle_only,
            )
            per_frame_ms.append((time.perf_counter() - t0) * 1000)
            frame_lines.append(lines)
        total_ms = (time.perf_counter() - t_start) * 1000

        # Merge adjacent same-text frames into segments
        segments = merge_frames(frame_lines)
        per_pass.append({
            "run": run + 1,
            "total_ms": round(total_ms, 2),
            "avg_per_frame_ms": round(sum(per_frame_ms) / len(per_frame_ms), 3),
            "segments": segments,
            "per_frame_ms": [round(x, 3) for x in per_frame_ms],
        })
        print(f"  run {run + 1}/{runs}: {len(segments)} segments, {total_ms:.1f}ms total, "
              f"{total_ms / len(per_frame_ms):.2f}ms/frame")
    return per_pass


def merge_frames(frame_lines):
    """Merge consecutive frames with identical text into segments."""
    segments = []
    current_text = None
    current_start = None
    current_end = None
    current_confs = []

    for i, lines in enumerate(frame_lines):
        text = " | ".join(l["text"] for l in lines) if lines else ""
        if text != current_text:
            if current_text and current_text.strip():
                segments.append({
                    "frame_range": [current_start, current_end],
                    "text": current_text,
                    "avg_confidence": round(sum(current_confs) / len(current_confs), 4),
                    "frame_count": current_end - current_start + 1,
                })
            current_text = text
            current_start = i
            current_end = i
            current_confs = [l["confidence"] for l in lines] if lines else []
        else:
            current_end = i
            if lines:
                current_confs.extend([l["confidence"] for l in lines])

    if current_text and current_text.strip():
        segments.append({
            "frame_range": [current_start, current_end],
            "text": current_text,
            "avg_confidence": round(sum(current_confs) / len(current_confs), 4),
            "frame_count": current_end - current_start + 1,
        })
    return segments


def summarize(per_pass):
    runs = len(per_pass)
    total_ms = [r["total_ms"] for r in per_pass]
    avg_frame_ms = [r["avg_per_frame_ms"] for r in per_pass]
    n_segs = [len(r["segments"]) for r in per_pass]

    summary = {
        "runs": runs,
        "total_ms_avg": round(statistics.mean(total_ms), 2),
        "total_ms_stddev": round(statistics.pstdev(total_ms), 2),
        "total_ms_min": round(min(total_ms), 2),
        "total_ms_max": round(max(total_ms), 2),
        "per_frame_ms_avg": round(statistics.mean(avg_frame_ms), 3),
        "per_frame_ms_stddev": round(statistics.pstdev(avg_frame_ms), 3),
        "segments_count_avg": round(statistics.mean(n_segs), 2),
        "segments_count_stddev": round(statistics.pstdev(n_segs), 3),
        "segments_count_min": min(n_segs),
        "segments_count_max": max(n_segs),
    }

    # Per-segment stability analysis
    all_texts_sets = [set(s["text"] for s in r["segments"]) for r in per_pass]
    union_texts = sorted(set().union(*all_texts_sets))

    seg_stats = {}
    for text in union_texts:
        confidences = []
        present_in = 0
        for r in per_pass:
            for s in r["segments"]:
                if s["text"] == text:
                    confidences.append(s["avg_confidence"])
                    present_in += 1
                    break
        seg_stats[text] = {
            "present_in_runs": present_in,
            "confidence_avg": round(statistics.mean(confidences), 4),
            "confidence_stddev": round(statistics.pstdev(confidences), 4) if len(confidences) > 1 else 0.0,
            "confidence_min": round(min(confidences), 4),
            "confidence_max": round(max(confidences), 4),
        }

    # Texts not present in every run
    missing_in_some = [t for t, v in seg_stats.items() if v["present_in_runs"] < runs]

    summary["unique_texts_total"] = len(union_texts)
    summary["unique_texts_common_all_runs"] = len(union_texts) - len(missing_in_some)
    summary["texts_missing_in_some_runs"] = len(missing_in_some)
    summary["_missing_texts_examples"] = missing_in_some[:20]
    summary["per_segment"] = seg_stats

    # Pooled per-frame timing (across all runs)
    all_frame_times = []
    for r in per_pass:
        all_frame_times.extend(r["per_frame_ms"])
    summary["pooled_per_frame_ms_count"] = len(all_frame_times)
    summary["pooled_per_frame_ms_avg"] = round(statistics.mean(all_frame_times), 3)
    summary["pooled_per_frame_ms_stddev"] = round(statistics.pstdev(all_frame_times), 3)
    if len(all_frame_times) > 1:
        summary["pooled_per_frame_ms_p50"] = round(statistics.quantiles(all_frame_times, n=2)[0], 3)
    summary["pooled_per_frame_ms_p95"] = round(sorted(all_frame_times)[int(len(all_frame_times) * 0.95)], 3)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--text-score", type=float, default=0.5)
    ap.add_argument("--subtitle-only", action="store_true", default=True)
    ap.add_argument("--frames-dir", type=str, default=str(DEFAULT_FRAMES),
                    help="directory with pre-extracted frame images (jpg/png)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="limit frames processed (0 = all)")
    ap.add_argument("--out", type=str,
                    default=str(OUT_DIR / "rapidocr-py-repeat-results.json"))
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    frames = get_frame_paths(frames_dir)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]

    print(f"RapidOCR (subtitle-py) repeat benchmark — {args.runs} runs on {len(frames)} frames")
    print(f"  frames: {frames_dir}")
    print(f"  text_score={args.text_score}, subtitle_only={args.subtitle_only}")
    print(f"  output: {args.out}")

    if not frames:
        print("ERROR: no frames found")
        sys.exit(1)

    t0 = time.perf_counter()
    per_pass = do_runs(frames, args.runs, args.text_score, args.subtitle_only)
    wall_ms = (time.perf_counter() - t0) * 1000

    summary = summarize(per_pass)
    out = {
        "description": f"RapidOCR (subtitle-py) repeat benchmark: {args.runs} runs on "
                       f"{len(frames)} frames from {frames_dir.name}, "
                       f"text_score={args.text_score}, subtitle_only={args.subtitle_only}",
        "args": vars(args),
        "frames_dir": str(frames_dir),
        "frames_total": len(frames),
        "wall_clock_total_ms": round(wall_ms, 2),
        "summary": summary,
        "per_run": per_pass,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nResults written to {args.out}")
    print(f"\n=== Summary ===")
    print(f"  runs:               {summary['runs']}")
    print(f"  total_ms:           {summary['total_ms_avg']} ± {summary['total_ms_stddev']} "
          f"(min {summary['total_ms_min']}, max {summary['total_ms_max']})")
    print(f"  per_frame_ms:       {summary['per_frame_ms_avg']} ± {summary['per_frame_ms_stddev']}")
    print(f"  segments per run:   {summary['segments_count_avg']} ± {summary['segments_count_stddev']} "
          f"(min {summary['segments_count_min']}, max {summary['segments_count_max']})")
    print(f"  unique texts:       {summary['unique_texts_total']} "
          f"(common to all runs: {summary['unique_texts_common_all_runs']}, "
          f"missing in some: {summary['texts_missing_in_some_runs']})")
    print(f"  pooled frame times: avg {summary['pooled_per_frame_ms_avg']}ms, "
          f"stddev {summary['pooled_per_frame_ms_stddev']}ms, "
          f"p95 {summary['pooled_per_frame_ms_p95']}ms")

    # Confidence summary
    confs_all = []
    for text, stats in summary["per_segment"].items():
        confs_all.append(stats["confidence_avg"])
    if confs_all:
        print(f"  avg segment confidence: {statistics.mean(confs_all):.4f}")


if __name__ == "__main__":
    main()
