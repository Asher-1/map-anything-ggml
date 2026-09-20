#!/usr/bin/env python3
"""PyTorch-CUDA latency baseline for the official VGGT-Omega (for the
ggml-vs-PyTorch comparison charts).

Usage:
  python3 benchmarks/bench_pytorch_baseline.py \
      --ckpt models/pytorch/vggt_omega_1b_512.pt \
      --frames /tmp/vggt_smoke/frames.bin --H 512 --W 512 --S 2 \
      --repeats 5 --warmup 2 --out benchmarks/results
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent / "third_party" / "vggt-omega-src"
sys.path.insert(0, str(SRC))
from vggt_omega.models import VGGTOmega  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the PyTorch baseline")
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    model = VGGTOmega().to(device).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd.get("state_dict", sd)))
    frames = np.fromfile(args.frames, dtype=np.float32).reshape(
        args.S, 3, args.H, args.W)
    images = torch.from_numpy(frames).unsqueeze(0).to(device)

    timings = []
    with torch.inference_mode():
        for i in range(args.warmup + args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(images)
            torch.cuda.synchronize()
            if i >= args.warmup:
                timings.append((time.perf_counter() - t0) * 1000.0)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

    res = {
        "backend": f"PyTorch-{device}",
        "ckpt": str(args.ckpt), "H": args.H, "W": args.W, "S": args.S,
        "repeats": args.repeats, "warmup": args.warmup,
        "e2e_ms": {"p50": statistics.median(timings),
                   "p95": sorted(timings)[int(0.95 * (len(timings) - 1))],
                   "mean": statistics.mean(timings)},
        "peak_vram_gb": peak_mem,
        "samples_ms": timings,
    }
    print(f"PyTorch-CUDA e2e P50 = {res['e2e_ms']['p50']:.1f} ms  "
          f"P95 = {res['e2e_ms']['p95']:.1f} ms  peak VRAM = {peak_mem:.2f} GB")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"pytorch_baseline_{args.H}x{args.W}x{args.S}.json"
    out_file.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
