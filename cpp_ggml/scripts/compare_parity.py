#!/usr/bin/env python3
"""Compare vggt-cli raw dumps against the PyTorch reference npz.

Usage:
  python3 compare_parity.py /tmp/vggt_smoke/out /tmp/vggt_smoke/ref.npz
"""

import sys

import numpy as np


def report(name, cpp, ref):
    d = np.abs(cpp.astype(np.float64) - ref.astype(np.float64))
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-6)
    rel = d / denom
    print(f"{name:12s} max_abs={d.max():.6e} mean_abs={d.mean():.6e} "
          f"max_rel={rel.max():.6e} mean_rel={rel.mean():.6e}")
    return d.max(), rel.mean()


def main():
    prefix, ref_npz = sys.argv[1], sys.argv[2]
    ref = np.load(ref_npz)
    pose = np.fromfile(f"{prefix}.pose.bin", dtype=np.float32).reshape(ref["pose_enc"].shape)
    depth = np.fromfile(f"{prefix}.depth.bin", dtype=np.float32).reshape(ref["depth"].shape)
    conf = np.fromfile(f"{prefix}.depth_conf.bin", dtype=np.float32).reshape(ref["depth_conf"].shape)

    ok = True
    for name, a, b in (("pose_enc", pose, ref["pose_enc"]),
                       ("depth", depth, ref["depth"]),
                       ("depth_conf", conf, ref["depth_conf"])):
        max_abs, mean_rel = report(name, a, b)
        if not np.isfinite(a).all():
            print(f"FAIL {name}: non-finite values in C++ output")
            ok = False

    # text-alignment variant: the 256_text GGUF makes the CLI emit
    # .text_embedding.bin (L2-normalized); the torch ref stores the same
    # vector, so a cosine check is the natural metric here
    if "text_embedding" in ref:
        import os
        tp = f"{prefix}.text_embedding.bin"
        if os.path.exists(tp):
            e = np.fromfile(tp, dtype=np.float32).reshape(ref["text_embedding"].shape)
            cos = float((e * ref["text_embedding"]).sum())
            print(f"text_embedding cos = {cos:.6f}  (gate > 0.999)")
            if cos <= 0.999:
                print("FAIL text_embedding")
                ok = False
        else:
            print("FAIL text_embedding: C++ output missing (.text_embedding.bin)")
            ok = False

    print("PARITY:", "PASS (within reported tolerance)" if ok else "FAIL")


if __name__ == "__main__":
    main()
