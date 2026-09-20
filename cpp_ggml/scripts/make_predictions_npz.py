#!/usr/bin/env python3
"""Package vggt-cli binary outputs into the OFFICIAL demo predictions.npz
contract (same keys as demo_gradio.run_model's np.savez).

Inputs : <prefix>.pose.bin (S,9) / .depth.bin (S,H,W) / .depth_conf.bin (S,H,W)
         (+ .meta.json for S/H/W when --s/--h/--w are not given)
Outputs: <prefix>.predictions.npz with keys
         pose_enc, depth, depth_conf, extrinsic (w2c, Sx3x4), intrinsic (Sx3x3),
         world_points_from_depth (S,H,W,3) — derived with the exact formulas
         used by the official demo (encoding_to_camera +
         unproject_depth_map_to_point_map).

Usage:
  python3 scripts/make_predictions_npz.py --prefix /tmp/mapggml/out
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="vggt-cli --out-prefix")
    ap.add_argument("--s", type=int, default=0)
    ap.add_argument("--h", type=int, default=0)
    ap.add_argument("--w", type=int, default=0)
    args = ap.parse_args()

    prefix = Path(args.prefix)
    s, h, w = args.s, args.h, args.w
    if not (s and h and w):
        meta = json.loads(Path(f"{prefix}.meta.json").read_text())
        s, h, w = s or meta["S"], h or meta["H"], w or meta["W"]

    pose_path = Path(f"{prefix}.pose.bin")
    if not pose_path.exists():
        _save_dust3r(prefix, s, h, w)
        return
    pose_n = pose_path.stat().st_size // 4
    per_view = pose_n // s
    if per_view == 16:
        _save_pi3_family(prefix, s, h, w, pose_n)
    else:
        _save_vggt(prefix, s, h, w)


def _save_dust3r(prefix, s, h, w):
    """dust3r pair-wise contract: NO pose (the model has no pose head;
    poses come from the out-of-network global-alignment optimizer). Both
    pointmaps live in view1's camera frame: s0 = head1 pts3d_in_self_view,
    s1 = head2 pts3d_in_other_view; conf is 1+exp (vmin=1); depth is the
    pts3d camera-frame z."""
    depth = np.fromfile(f"{prefix}.depth.bin", dtype=np.float32).reshape(s, h, w)
    conf = np.fromfile(f"{prefix}.conf.bin", dtype=np.float32).reshape(s, h, w)
    local = np.fromfile(f"{prefix}.local_points.bin",
                        dtype=np.float32).reshape(s, h, w, 3)
    out = prefix.parent / (prefix.name + ".predictions.npz")
    keys = ["depth", "conf", "local_points"]
    arrays = dict(depth=depth[..., None], conf=conf, local_points=local)
    if (p := Path(f"{prefix}.points.bin")).exists():
        arrays["world_points"] = np.fromfile(
            p, dtype=np.float32).reshape(s, h, w, 3)
        keys.append("world_points")
    np.savez(out, **arrays)
    print(f"wrote {out} keys: {', '.join(keys)} (no pose: dust3r has no "
          f"pose head)")


def _save_vggt(prefix, s, h, w):
    """Official demo contract (pose_enc = t|quat|fov, camera-from-world)."""
    pose = np.fromfile(f"{prefix}.pose.bin", dtype=np.float32).reshape(s, 9)
    depth = np.fromfile(f"{prefix}.depth.bin", dtype=np.float32).reshape(s, h, w)
    conf = np.fromfile(f"{prefix}.depth_conf.bin", dtype=np.float32).reshape(s, h, w)

    # Official pose_enc layout: [trans(3) | quat(4, xyzw scalar-last) |
    # fov_h | fov_w], camera-from-world. Decoding matches the official
    # encoding_to_camera: fy = (H/2)/tan(fov_h), fx = (W/2)/tan(fov_w),
    # principal point at the image center, extrinsic (S,3,4) w2c.
    ext = np.zeros((s, 3, 4), dtype=np.float32)
    K = np.zeros((s, 3, 3), dtype=np.float32)
    for i in range(s):
        q = pose[i, 3:7] / np.linalg.norm(pose[i, 3:7])
        x, y, z, wq = q
        ext[i, 0, 0] = 1 - 2 * (y * y + z * z)
        ext[i, 0, 1] = 2 * (x * y - z * wq)
        ext[i, 0, 2] = 2 * (x * z + y * wq)
        ext[i, 1, 0] = 2 * (x * y + z * wq)
        ext[i, 1, 1] = 1 - 2 * (x * x + z * z)
        ext[i, 1, 2] = 2 * (y * z - x * wq)
        ext[i, 2, 0] = 2 * (x * z - y * wq)
        ext[i, 2, 1] = 2 * (y * z + x * wq)
        ext[i, 2, 2] = 1 - 2 * (x * x + y * y)
        ext[i, :3, 3] = pose[i, :3]
        K[i, 0, 0] = (w / 2) / np.tan(pose[i, 8] / 2)
        K[i, 1, 1] = (h / 2) / np.tan(pose[i, 7] / 2)
        K[i, 0, 2] = w / 2
        K[i, 1, 2] = h / 2
        K[i, 2, 2] = 1.0

    # official unproject_depth_map_to_point_map: world = R^T (p_cam - t)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = np.broadcast_to(xx[None], (s, h, w)).astype(np.float64)
    y = np.broadcast_to(yy[None], (s, h, w)).astype(np.float64)
    d = depth.astype(np.float64)
    fx = K[:, 0, 0][:, None, None].astype(np.float64)
    fy = K[:, 1, 1][:, None, None].astype(np.float64)
    cx = K[:, 0, 2][:, None, None].astype(np.float64)
    cy = K[:, 1, 2][:, None, None].astype(np.float64)
    cam = np.stack([(x - cx) / fx * d, (y - cy) / fy * d, d], axis=-1)
    rot = ext[:, :3, :3].astype(np.float64)
    tr = ext[:, :3, 3].astype(np.float64)
    world = np.einsum("sij,shwj->shwi", np.transpose(rot, (0, 2, 1)),
                      cam - tr[:, None, None, :])

    out = prefix.parent / (prefix.name + ".predictions.npz")
    np.savez(out, pose_enc=pose, depth=depth[..., None], depth_conf=conf,
             extrinsic=ext, intrinsic=K, world_points_from_depth=world)
    print(f"wrote {out} keys: pose_enc, depth, depth_conf, extrinsic, "
          f"intrinsic, world_points_from_depth")


def _save_pi3_family(prefix, s, h, w, pose_n):
    """pi3/pi3x/mapanything contract: .pose.bin holds ROW-MAJOR
    camera-to-world 4x4 (translations already in metric scale for pi3x /
    mapanything; raw scale-invariant for pi3), local_points are metric
    camera-frame points (pi3/pi3x) or unit ray dirs (mapanything), .points.bin
    is the world-frame cloud, .scale.bin the metric factor (absent for pi3)."""
    pose = np.fromfile(f"{prefix}.pose.bin", dtype=np.float32).reshape(s, 4, 4)
    depth = np.fromfile(f"{prefix}.depth.bin", dtype=np.float32).reshape(s, h, w)
    conf = np.fromfile(f"{prefix}.conf.bin", dtype=np.float32).reshape(s, h, w)
    local = np.fromfile(f"{prefix}.local_points.bin",
                        dtype=np.float32).reshape(s, h, w, 3)
    out = prefix.parent / (prefix.name + ".predictions.npz")
    keys = ["pose_c2w", "extrinsic", "depth", "conf", "local_points"]
    arrays = dict(pose_c2w=pose, depth=depth[..., None], conf=conf,
                  local_points=local)
    # w2c extrinsic (S,3,4) from the row-major c2w
    ext = np.zeros((s, 3, 4), dtype=np.float32)
    for i in range(s):
        ext[i] = np.linalg.inv(pose[i].astype(np.float64))[:3]
    arrays["extrinsic"] = ext
    if (p := Path(f"{prefix}.points.bin")).exists():
        arrays["world_points"] = np.fromfile(
            p, dtype=np.float32).reshape(s, h, w, 3)
        keys.append("world_points")
    if (p := Path(f"{prefix}.scale.bin")).exists():
        arrays["metric_scaling_factor"] = float(np.fromfile(p, dtype=np.float32)[0])
        keys.append("metric_scaling_factor")
    np.savez(out, **arrays)
    print(f"wrote {out} keys: {', '.join(keys)}")


if __name__ == "__main__":
    main()
