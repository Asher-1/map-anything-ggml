#!/usr/bin/env python3
"""End-to-end accuracy benchmark: run every quant through the C++ CLI and
compare against the PyTorch reference (ref.npz from dump_torch_stages.py).

Usage:
  python3 benchmarks/bench_e2e.py \
      --cli build-cpu/bin/vggt-cli --models-dir models/gguf \
      --model vggt-omega-1b-512 --quants "f32 f16 q8_0 q5_K q4_K" \
      --ref /tmp/vggt_stages/ref.npz --frames /tmp/vggt_smoke/frames.bin \
      --H 512 --W 512 --S 2 --out benchmarks/results
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def metrics(pose_out, depth_out, ref_pose, ref_depth):
    pose_err = float(np.abs(pose_out - ref_pose).max())
    r = ref_depth
    d = np.abs(depth_out - r)
    rel = d / np.maximum(np.abs(r), 0.05)
    return {"pose_max_abs": pose_err,
            "depth_median_rel": float(np.median(rel)),
            "depth_p90_rel": float(np.percentile(rel, 90)),
            "nan": bool(np.isnan(depth_out).any() or np.isnan(pose_out).any())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--model", required=True, help="gguf stem")
    ap.add_argument("--quants", default="f32 f16 q8_0 q5_K q4_K")
    ap.add_argument("--ref", required=True, help="ref.npz from dump_torch_stages.py")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    ref = np.load(args.ref)
    ref_pose = ref["pose_enc"].reshape(args.S, 9)
    ref_depth = ref["depth"].reshape(args.S, args.H, args.W)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"model": args.model, "H": args.H, "W": args.W, "S": args.S,
               "ref": str(args.ref), "entries": {}}

    print(f"{'quant':8s} {'pose max_abs':>13s} {'depth med_rel':>14s} "
          f"{'depth p90_rel':>14s} {'nan':>4s}")
    for q in args.quants.replace(",", " ").split():
        gguf = Path(args.models_dir) / f"{args.model}-{q}.gguf"
        if not gguf.exists():
            print(f"{q:8s}  -- missing {gguf.name}, skipped")
            continue
        prefix = f"/tmp/mapggml_e2e_{args.model}_{q}"
        proc = subprocess.run(
            [str(args.cli), "--model", str(gguf), "--bin", str(args.frames),
             "--H", str(args.H), "--W", str(args.W), "--S", str(args.S),
             "--out-prefix", prefix],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"CLI failed for {q}:\n{proc.stderr[-2000:]}")
        pose = np.fromfile(f"{prefix}.pose.bin", dtype=np.float32).reshape(args.S, 9)
        depth = np.fromfile(f"{prefix}.depth.bin", dtype=np.float32).reshape(
            args.S, args.H, args.W)
        m = metrics(pose, depth, ref_pose, ref_depth)
        results["entries"][q] = m
        print(f"{q:8s} {m['pose_max_abs']:>13.4f} {m['depth_median_rel']:>14.4f} "
              f"{m['depth_p90_rel']:>14.4f} {str(m['nan']):>4s}")

    out_file = out_dir / f"e2e_{args.model}_{args.H}x{args.W}x{args.S}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
