#!/usr/bin/env python3
"""Render a fused cloud to top-down + two side projections (matplotlib)
for quick visual QA of chunk alignment (walls should coincide, no ghosts).

Usage:
  python3 scripts/preview_ply.py cloud.ply --out preview.png [--title str]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt = f.readline().strip()
        n = 0
        props = []
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                props.append(line.split()[-1].decode())
            elif line == b"end_header":
                break
        dt = np.dtype([(p, "<f4" if p in ("x", "y", "z") else "u1")
                       for p in props])
        data = np.fromfile(f, dtype=dt, count=n)
    pts = np.stack([data["x"], data["y"], data["z"]], -1)
    cols = np.stack([data["red"], data["green"], data["blue"]], -1) / 255.0
    return pts, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    out = a.out or str(Path(a.ply).with_suffix(".preview.png"))
    pts, cols = read_ply(a.ply)
    # GT-ish orientation: courtyard z is roughly up in the WAI frame; project
    # onto the two horizontal axes for top view and z for the side views.
    zc = pts[:, 2]
    up = np.argsort(zc)[int(len(zc) * 0.05):int(len(zc) * 0.95)]
    lo, hi = np.percentile(zc, [2, 98])
    span = max(hi - lo, 1e-6)
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.5))
    for ax, (i, j, ti) in zip(axs, [(0, 1, "top (x-y)"),
                                    (0, 2, "side (x-z)"),
                                    (1, 2, "side (y-z)")]):
        s = slice(None)
        ax.scatter(pts[s, i], pts[s, j], c=cols[s], s=0.05, lw=0,
                   alpha=0.5)
        ax.set_title(ti)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"{Path(a.ply).name}  {a.title}  ({len(pts):,} pts)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
