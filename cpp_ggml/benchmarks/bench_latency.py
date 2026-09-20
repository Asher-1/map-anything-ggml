#!/usr/bin/env python3
"""Latency benchmark for vggt-cli across quant formats.

Runs the CLI ONCE per quant with --warmup/--repeats so all iterations happen
in a single process: ggml's CUDA graph replay only engages after two stable
in-process computes, so cold per-iteration processes would measure pure
kernel-launch overhead (~60 ms) instead of steady-state latency.

Usage:
  python3 benchmarks/bench_latency.py \
      --cli build-cpu/bin/vggt-cli --models-dir models/gguf \
      --model vggt-omega-1b-512 --quants "f16 q8_0 q5_K" \
      --frames /tmp/vggt_smoke/frames.bin --H 512 --W 512 --S 2 \
      --repeats 5 --warmup 2 --out benchmarks/results
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def run_cli(cli, model_path, frames, H, W, S, warmup, repeats):
    proc = subprocess.run(
        [str(cli), "--model", str(model_path), "--bin", str(frames),
         "--H", str(H), "--W", str(W), "--S", str(S),
         "--out-prefix", "/tmp/mapggml_bench", "--timing",
         "--warmup", str(warmup), "--repeats", str(repeats)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    m = re.search(r"\{\"backend\".*\}", proc.stderr)
    if not m:
        raise RuntimeError(f"no timing JSON in stderr:\n{proc.stderr[-2000:]}")
    return json.loads(m.group(0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--model", required=True, help="gguf stem, e.g. vggt-omega-1b-512")
    ap.add_argument("--quants", default="f16")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    quants = args.quants.replace(",", " ").split()

    results = {"cli": str(args.cli), "model": args.model, "H": args.H,
               "W": args.W, "S": args.S, "repeats": args.repeats,
               "warmup": args.warmup, "entries": {}}
    print(f"{'quant':8s} {'infer P50':>10s} {'P95':>10s} {'mean':>10s} "
          f"{'stage1':>10s} {'stage2':>10s}")
    for q in quants:
        gguf = Path(args.models_dir) / f"{args.model}-{q}.gguf"
        if not gguf.exists():
            print(f"{q:8s}  -- missing {gguf.name}, skipped")
            continue
        # the frames file may contain more views than S (throughput scans
        # reuse an S_max buffer); CLI validates size, so slice per run
        all_frames = np.fromfile(args.frames, dtype=np.float32)
        per_frame = 3 * args.H * args.W
        if len(all_frames) < args.S * per_frame:
            raise RuntimeError(
                f"frames file too small: {len(all_frames)} < {args.S * per_frame}")
        frames_path = str(args.frames) if len(all_frames) == args.S * per_frame \
            else str(Path(args.frames).parent /
                     (Path(args.frames).stem + f".s{args.S}.bin"))
        if len(all_frames) != args.S * per_frame:
            all_frames[:args.S * per_frame].tofile(frames_path)
        timings = []
        # single process: warmup iterations engage CUDA graph replay, the
        # timed repeats report steady-state per-iteration latency
        t = run_cli(args.cli, gguf, frames_path, args.H, args.W, args.S,
                    args.warmup, args.repeats)
        timings.append(t)
        tm = t["timing_ms"]
        # the CLI only emits inference_p50/p95/mean when repeats > 1; a
        # single-shot run (--repeats 1) reports inference_total instead
        if "inference_p50" in tm:
            infer = {"p50": tm["inference_p50"],
                     "p95": tm["inference_p95"],
                     "mean": tm["inference_mean"]}
        else:
            tot = float(tm.get("inference_total", 0.0))
            infer = {"p50": tot, "p95": tot, "mean": tot}
        entry = {
            "backend": t["backend"],
            "inference_ms": infer,
            "stage_ms": {"stage1_backbone": tm.get("stage1_p50", 0.0),
                         "stage2_aggregator": tm.get("stage2_p50", 0.0),
                         "stage3_heads": 0.0},
            "iterations_ms": tm.get("iterations_ms", []),
            "samples": timings,
        }
        results["entries"][q] = entry
        print(f"{q:8s} {entry['inference_ms']['p50']:>10.1f} "
              f"{entry['inference_ms']['p95']:>10.1f} "
              f"{entry['inference_ms']['mean']:>10.1f} "
              f"{entry['stage_ms']['stage1_backbone']:>10.1f} "
              f"{entry['stage_ms']['stage2_aggregator']:>10.1f}")

    out_file = out_dir / (f"latency_{args.model}_{args.H}x{args.W}x{args.S}" +
                          (f"_{results['entries'][next(iter(results['entries']))]['backend']}"
                           if results['entries'] else "") + ".json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
