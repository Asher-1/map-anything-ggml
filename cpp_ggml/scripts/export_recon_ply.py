#!/usr/bin/env python3
"""Export a reconstruction point cloud (world points + per-pixel color)
to a binary PLY for human inspection in any viewer.

Sources:
  --npz  predictions.npz with the OFFICIAL demo keys
         (world_points_from_depth (S,H,W,3) + images (S,H,W,3) uint8)
  --npy  cpp-side world_points_from_depth.npy (from compare_official_demo.py);
         colors are then taken from --npz's images (same preprocessed frames)

All views are concatenated into one cloud; an optional --conf-npy /
npz["depth_conf"] mask keeps points above a confidence floor.
"""
import argparse
from pathlib import Path

import numpy as np


def write_ply(path, pts, colors):
    n = len(pts)
    assert len(colors) == n
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        cloud = np.zeros(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        cloud["xyz"] = pts.astype(np.float32)
        cloud["rgb"] = colors.astype(np.uint8)
        f.write(cloud.tobytes())
    print(f"wrote {path}: {n:,} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="predictions.npz (official keys; supplies images and,"
                         " unless --npy, world_points_from_depth)")
    ap.add_argument("--npy", default="",
                    help="cpp world_points_from_depth.npy (uses --npz images)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf-min", type=float, default=0.0,
                    help="optional depth_conf floor (0 = keep everything)")
    a = ap.parse_args()

    z = np.load(a.npz)
    colors = z["images"]
    if colors.ndim == 4 and colors.shape[1] == 3:
        colors = colors.transpose(0, 2, 3, 1)  # torch CHW -> HWC
    if colors.dtype != np.uint8:
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    world = (np.load(a.npy) if a.npy
             else z["world_points_from_depth"])
    world = np.asarray(world, dtype=np.float32)
    colors = np.broadcast_to(colors, world.shape[:-1] + (3,))

    pts = world.reshape(-1, 3)
    cols = np.broadcast_to(colors, world.shape[:-1] + (3,)).reshape(-1, 3)
    if a.conf_min > 0:
        conf = (np.load(a.conf_npy) if getattr(a, "conf_npy", "") and a.npy
                else z["depth_conf"]).reshape(-1)
        keep = conf >= a.conf_min
        pts, cols = pts[keep], cols[keep]
    write_ply(a.out, pts, cols)


if __name__ == "__main__":
    main()
