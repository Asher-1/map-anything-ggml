#!/usr/bin/env python3
"""Regression matrix: run (backend × model × quant) through vggt-cli and
assert parity against a fresh PyTorch f32 reference.

This is the executable form of the acceptance table in RESULTS.md. It is the
gate that would have caught the S>=2 misalignment bug on day one.

Usage:
  python3 tests/test_matrix.py                       # CPU × all quants
  python3 tests/test_matrix.py --backends cuda vulkan
  python3 tests/test_matrix.py --models 512 416 text
Exits non-zero if any cell fails its tolerance.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CPP = Path(__file__).resolve().parent.parent
MODEL_RES = {"512": 512, "416": 416, "text": 256}
MODEL_CKPT = {
    "512": "vggt-omega/vggt_omega_1b_512.pt",
    "416": "vggt-omega/vggt_omega_1b_416_reproduce.pt",
    "text": "vggt-omega/vggt_omega_1b_256_text.pt",
}
MODEL_GGUF = {
    "512": "vggt-omega-1b-512",
    "416": "vggt-omega-1b-416-reproduce",
    "text": "vggt-omega-1b-256-text",
}
QUANTS = ["f16", "q8_0", "q5_K"]
# gate thresholds (RESULTS.md acceptance table)
POSE_TOL = 0.05
DEPTH_TOL = 0.08


def torch_ref(frames_path, model_key, S, H, W, out_dir):
    """(Re)generate the PyTorch f32 reference for the given frames/model."""
    ref_npz = out_dir / f"ref_{model_key}_{S}.npz"
    ckpt = CPP / "models" / "pytorch" / MODEL_CKPT[model_key]
    if not ref_npz.exists():
        subprocess.run(
            [sys.executable, str(CPP / "scripts" / "dump_torch_stages.py"),
             str(ckpt), str(frames_path), str(H), str(W), str(S), str(out_dir)],
            check=True, capture_output=True, text=True,
            cwd=str(out_dir))
        # dump_torch_stages writes ref.npz into out_dir
        (out_dir / "ref.npz").rename(ref_npz)
    return ref_npz


def run_cell(cli, model_key, quant, frames_path, S, H, W, prefix):
    gguf = CPP / "models" / "gguf" / f"{MODEL_GGUF[model_key]}-{quant}.gguf"
    if not gguf.exists():
        return None
    proc = subprocess.run(
        [str(cli), "--model", str(gguf), "--bin", str(frames_path),
         "--H", str(H), "--W", str(W), "--S", str(S), "--out-prefix", prefix],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI failed for {model_key}/{quant}:\n{proc.stderr[-1500:]}")
    pose = np.fromfile(f"{prefix}.pose.bin", dtype=np.float32).reshape(S, 9)
    depth = np.fromfile(f"{prefix}.depth.bin", dtype=np.float32).reshape(S, H, W)
    return pose, depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*", default=["cpu"])
    ap.add_argument("--models", nargs="*", default=["512"])
    ap.add_argument("--quants", nargs="*", default=QUANTS)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--build-root", default=str(CPP))
    ap.add_argument("--work-dir", default="/tmp/mapggml_test_matrix")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    failures = []
    rows = []
    for model_key in args.models:
        res = MODEL_RES[model_key]
        H = W = res
        frames = work / f"frames_{model_key}_{args.S}.bin"
        if not frames.exists() or frames.stat().st_size != args.S * 3 * H * W * 4:
            rng.random((args.S, 3, H, W), dtype=np.float32).astype(
                np.float32).tofile(frames)
        ref_npz = torch_ref(frames, model_key, args.S, H, W, work)
        ref = np.load(ref_npz)
        for backend in args.backends:
            build = Path(args.build_root) / f"build-{backend}"
            cli = build / "bin" / "vggt-cli"
            if not cli.exists():
                print(f"SKIP {backend}: no {cli}")
                continue
            for quant in args.quants:
                prefix = str(work / f"out_{backend}_{model_key}_{quant}")
                try:
                    got = run_cell(cli, model_key, quant, frames, args.S, H, W,
                                   prefix)
                except RuntimeError as e:
                    failures.append(f"{backend}/{model_key}/{quant}: {e}")
                    print(f"FAIL {backend}/{model_key}/{quant}: runtime error")
                    continue
                if got is None:
                    print(f"SKIP {backend}/{model_key}/{quant}: no gguf")
                    continue
                pose, depth = got
                pe = float(np.abs(pose - ref["pose_enc"]).max())
                rel = np.abs(depth - ref["depth"]) / np.maximum(
                    np.abs(ref["depth"]), 0.05)
                dr = float(np.median(rel))
                nan = bool(np.isnan(depth).any())
                ok = pe < POSE_TOL and dr < DEPTH_TOL and not nan
                rows.append((backend, model_key, quant, pe, dr, ok))
                status = "PASS" if ok else "FAIL"
                print(f"{status} {backend:6s}/{model_key:3s}/{quant:5s} "
                      f"pose={pe:.4f} depth_med={dr:.4f}")
                if not ok:
                    failures.append(
                        f"{backend}/{model_key}/{quant}: pose={pe:.4f} "
                        f"depth={dr:.4f} nan={nan}")

    print("\n=== summary ===")
    for r in rows:
        print(f"{'PASS' if r[5] else 'FAIL'} {r[0]:6s} {r[1]:4s} {r[2]:5s} "
              f"pose={r[3]:.4f} depth={r[4]:.4f}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        sys.exit(1)
    print("\nALL CELLS PASS")


if __name__ == "__main__":
    main()
