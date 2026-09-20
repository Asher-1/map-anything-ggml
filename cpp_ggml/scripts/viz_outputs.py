#!/usr/bin/env python3
"""Visualize vggt-cli outputs (<prefix>.depth.bin / .depth_conf.bin) as a
side-by-side color map PNG (<prefix>.depth_vis.png). Optional nicety used by
run_mapggml.sh infer; exits non-zero quietly when matplotlib is missing."""
import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="output prefix of vggt-cli")
    ap.add_argument("--out", default="", help="output png (default <prefix>.depth_vis.png)")
    args = ap.parse_args()

    meta_p = Path(f"{args.prefix}.meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    S, H, W = meta.get("S", 0), meta.get("H", 0), meta.get("W", 0)
    depth = np.fromfile(f"{args.prefix}.depth.bin", dtype=np.float32)
    if H <= 0 or W <= 0:
        raise SystemExit("meta.json missing S/H/W; cannot reshape depth.bin")
    depth = depth.reshape(S, H, W)
    conf = np.fromfile(f"{args.prefix}.depth_conf.bin", dtype=np.float32).reshape(S, H, W)

    n = S
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False)
    for s in range(n):
        d = depth[s]
        dmin, dmax = np.percentile(d, 2), np.percentile(d, 98)
        im = axes[0][s].imshow(np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1),
                               cmap="magma")
        axes[0][s].set_title(f"depth view {s}")
        axes[0][s].axis("off")
        fig.colorbar(im, ax=axes[0][s], fraction=0.046)
        c = conf[s]
        cmin, cmax = np.percentile(c, 2), np.percentile(c, 98)
        axes[1][s].imshow(np.clip((c - cmin) / max(cmax - cmin, 1e-6), 0, 1),
                          cmap="viridis")
        axes[1][s].set_title(f"depth_conf view {s}")
        axes[1][s].axis("off")
    out = args.out or f"{args.prefix}.depth_vis.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
