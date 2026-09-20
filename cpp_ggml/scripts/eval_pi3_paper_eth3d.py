#!/usr/bin/env python3
"""pi3 paper ETH3D point-map protocol (paper Table 3) reproduction.

Protocol (pi3 paper, following CUT3R's mv_recon eval):
  * per scene: the FULL WAI frame sequence; keyframes = frames[::stride=5];
  * the model sees all keyframes at an aspect-preserving /14 resolution
    (width 518, height rounded to /14);
  * predictions = global point maps (pi3 is scale-invariant);
  * GT points = the GT depth (native EXR) area-downsampled to the model
    resolution and back-projected with the scaled intrinsics, per keyframe,
    giving a per-pixel correspondence with the predictions;
  * alignment: Umeyama Sim(3) pred->gt on the per-pixel correspondences,
    then Open3D point-to-point ICP refinement (threshold 0.1, as CUT3R);
  * metrics over the whole scene cloud: Acc/Comp = KDTree distance
    mean/median; NC = |normal dot| mean/median (o3d estimate_normals);
  * scenes averaged.

Usage:
  PYTHONPATH=/tmp/torch_cuda_lib:<repo> python3 scripts/eval_pi3_paper_eth3d.py \
      --data-root <...>/eth3d --side torch            # official torch pi3
  python3 scripts/eval_pi3_paper_eth3d.py --data-root <...>/eth3d \
      --side cpp --cli build-cuda/bin/vggt-cli --gguf models/gguf/pi3-f16.gguf
  ... --side paper  # print the paper's Table-3 reference numbers
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
CPP = Path(__file__).resolve().parent.parent

PAPER_ETH3D = {  # pi3 paper Table 3, "pi3 (Ours)" row
    "acc_mean": 0.194, "acc_med": 0.131,
    "comp_mean": 0.210, "comp_med": 0.128,
    "nc_mean": 0.883, "nc_med": 0.969,
}
STRIDE = 5
MODEL_W = 518


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Least-squares Sim(3): dst ~= s * R @ src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c.T @ s_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (s_c ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def load_scene(data_root: Path, scene: str):
    meta = json.loads((data_root / scene / "scene_meta.json").read_text())
    frames = sorted(meta["frames"], key=lambda f: f["frame_name"])
    return frames


def pick_resolution(h, w):
    """aspect-preserving width-518, /14 height."""
    hh = max(14, int(round(h * MODEL_W / w / 14.0)) * 14)
    return hh, MODEL_W


def backproject(depth, fl_x, fl_y, cx, cy, valid):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) / fl_x * depth
    y = (v - cy) / fl_y * depth
    pts = np.stack([x, y, depth], axis=-1)   # full grid; masking at the caller
    pts[~valid] = np.nan                     # NaN (not 0) so normals stay clean
    return pts


def grid_normals(pts):
    """analytic normals of a regular (H, W, 3) point grid via central
    differences (far more accurate than KD-tree estimation on downsampled
    depth)."""
    dz = np.gradient(pts, axis=0)
    dx = np.gradient(pts, axis=1)
    n = np.cross(dz, dx)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    return n


def gt_cloud_for_scene(frames, hh, ww, pred_shape=None):
    """Per keyframe: GT points at the MODEL resolution (camera frame, for the
    per-pixel Umeyama correspondence) plus NATIVE-resolution analytic
    normals resampled to the model grid (world transform in scene_metrics)."""
    import cv2
    import imageio.v3 as iio

    pts_cam_all, nrm_cam_all, c2ws = [], [], []
    for f in frames:
        depth_native = iio.imread(str(f["depth_abs"])).astype(np.float32)
        depth_native = depth_native[..., 0] if depth_native.ndim == 3 else depth_native
        valid_n = depth_native > 0
        # analytic normals at the NATIVE resolution
        pts_n = backproject(depth_native, f["fl_x"], f["fl_y"],
                            f["cx"], f["cy"], valid_n)
        n_n = grid_normals(pts_n)
        # area-downsample depth to the model grid (per-pixel correspondence)
        depth = cv2.resize(depth_native, (ww, hh), interpolation=cv2.INTER_AREA)
        valid = depth > 0
        sh = hh / f["h"]
        sw = ww / f["w"]
        pc = backproject(depth, f["fl_x"] * sw, f["fl_y"] * sh,
                         f["cx"] * sw, f["cy"] * sh, valid)
        # nearest-resample the native normals onto the model grid
        n_m = cv2.resize(n_n, (ww, hh), interpolation=cv2.INTER_NEAREST)
        pts_cam_all.append(pc)
        nrm_cam_all.append(n_m)
        c2ws.append(np.asarray(f["transform_matrix"], dtype=np.float64))
    return pts_cam_all, nrm_cam_all, c2ws


def torch_predictions(frames, hh, ww, stride, arch="pi3"):
    CPPD = CPP / "third_party" / "pi3-src"
    sys.path.insert(0, str(CPPD))
    import cv2
    import torch
    if arch == "pi3x":
        from pi3.models.pi3x import Pi3X as Model
        ckpt = CPP / "models/pytorch/pi3x"
    else:
        from pi3.models.pi3 import Pi3 as Model
        ckpt = CPP / "models/pytorch/pi3"

    model = Model.from_pretrained(str(ckpt)).cuda().eval()
    kf = frames[::stride]
    imgs = []
    for f in kf:
        im = cv2.imread(str(f["image_abs"]), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (ww, hh), interpolation=cv2.INTER_AREA)
        imgs.append(torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1))
    imgs = torch.stack(imgs).cuda().float() / 255.0        # (S,3,H,W) in [0,1]
    with torch.no_grad():
        res = model(imgs[None])                             # (1,S,3,H,W); the model applies its ImageNet buffers internally
    pts = res["points"][0].float().cpu().numpy()            # (S,H,W,3)
    conf = res["conf"][0].float().cpu().numpy()[..., 0]     # (S,H,W)
    return pts, conf


def cpp_predictions(frames, hh, ww, stride, cli, gguf):
    import cv2
    kf = frames[::stride]
    imgs = []
    for f in kf:
        im = cv2.imread(str(f["image_abs"]), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (ww, hh), interpolation=cv2.INTER_AREA)
        imgs.append(im[..., ::-1].copy())
    arr = np.stack(imgs).astype(np.float32) / 255.0         # (S,3?,H,W) -> chw
    arr = arr.transpose(0, 3, 1, 2)
    S = arr.shape[0]
    with tempfile.TemporaryDirectory() as td:
        arr.tofile(f"{td}/frames.bin")
        prefix = f"{td}/out"
        proc = subprocess.run(
            [cli, "--model", gguf, "--bin", f"{td}/frames.bin",
             "--H", str(hh), "--W", str(ww), "--S", str(S),
             "--out-prefix", prefix], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"CLI failed:\n{proc.stderr[-600:]}")
        pts = np.fromfile(f"{prefix}.points.bin", dtype=np.float32).reshape(S, hh, ww, 3)
        conf = np.fromfile(f"{prefix}.conf.bin", dtype=np.float32).reshape(S, hh, ww)
    return pts, conf


def scene_metrics(pred_pts, pred_conf, gt_pc_cam, gt_nm_cam, c2ws, use_conf_mask):
    import open3d as o3d
    from scipy.spatial import cKDTree

    # pi3 predicts scale-INVARIANT local point maps per view (the global
    # points therefore carry a per-view arbitrary scale, unlike the poses
    # which share one consistent frame).  The scale-invariant point-map
    # protocol aligns EACH view's prediction to its own GT with an
    # independent Umeyama Sim(3) + ICP, then merges the aligned views.
    pcd_p = o3d.geometry.PointCloud()
    pcd_g = o3d.geometry.PointCloud()
    for s_i, (pc_cam, n_cam, c2w) in enumerate(zip(gt_pc_cam, gt_nm_cam, c2ws)):
        Rgw = c2w[:3, :3]
        gt_world = pc_cam @ Rgw.T + c2w[:3, 3]
        pred = pred_pts[s_i]
        valid = pc_cam[..., 2] > 0                  # GT depth validity
        # pi3's conf head approximates a binary validity mask (the Pi3X
        # release notes state this explicitly), so keep sigmoid(conf) > 0.5,
        # i.e. raw logits > 0.
        valid = valid & (pred_conf[s_i] > 0.0)
        if use_conf_mask:  # extra robustness option: drop the low quartile
            conf = pred_conf[s_i]
            q = np.quantile(conf[valid], 0.25) if valid.any() else 0.0
            valid = valid & (conf > q)
        p = pred.reshape(-1, 3)[valid.reshape(-1)]
        g = gt_world.reshape(-1, 3)[valid.reshape(-1)]
        # analytic normals; both rotated into the world frame (pred's view
        # normals from the point grid, GT's resampled from native resolution).
        # Acc/Comp use the FULL point sets; normals that are invalid (NaN from
        # depth borders / INTER_NEAREST sampling of invalid native pixels)
        # are zeroed here and masked out of the NC metric only.
        pn = grid_normals(pred).reshape(-1, 3)[valid.reshape(-1)]
        pn = pn @ Rgw.T
        gn = n_cam.reshape(-1, 3)[valid.reshape(-1)] @ Rgw.T
        pn[~(np.isfinite(pn).all(-1) & (np.linalg.norm(np.nan_to_num(pn), axis=-1) > 0.5))] = 0.0
        gn[~(np.isfinite(gn).all(-1) & (np.linalg.norm(np.nan_to_num(gn), axis=-1) > 0.5))] = 0.0
        fin = np.isfinite(p).all(-1) & np.isfinite(g).all(-1)
        p, g, pn, gn = p[fin], g[fin], pn[fin], gn[fin]
        if len(p) < 100:
            continue
        scale, R, t = umeyama_sim3(p, g)
        p = (scale * (R @ p.T)).T + t
        pn = (R @ pn.T).T
        pv = o3d.geometry.PointCloud()
        pv.points = o3d.utility.Vector3dVector(p)
        pv.normals = o3d.utility.Vector3dVector(pn)
        gv = o3d.geometry.PointCloud()
        gv.points = o3d.utility.Vector3dVector(g)
        gv.normals = o3d.utility.Vector3dVector(gn)
        icp = o3d.pipelines.registration.registration_icp(
            pv, gv, 0.1, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        pv = pv.transform(icp.transformation)   # rotates normals too
        pcd_p += pv
        pcd_g += gv

    pa = np.asarray(pcd_p.points)
    ga = np.asarray(pcd_g.points)
    na = np.asarray(pcd_p.normals)
    ng = np.asarray(pcd_g.normals)

    d_gt_of_pred, idx1 = cKDTree(ga).query(pa, workers=-1)
    d_pred_of_gt, idx2 = cKDTree(pa).query(ga, workers=-1)
    # NC only over pixels with valid normals on BOTH sides
    ok1 = (np.linalg.norm(na, axis=-1) > 0.5) & (np.linalg.norm(ng[idx1], axis=-1) > 0.5)
    ok2 = (np.linalg.norm(ng, axis=-1) > 0.5) & (np.linalg.norm(na[idx2], axis=-1) > 0.5)
    nc1 = np.abs((ng[idx1][ok1] * na[ok1]).sum(-1))
    nc2 = np.abs((na[idx2][ok2] * ng[ok2]).sum(-1))
    return {
        "acc_mean": float(d_gt_of_pred.mean()), "acc_med": float(np.median(d_gt_of_pred)),
        "comp_mean": float(d_pred_of_gt.mean()), "comp_med": float(np.median(d_pred_of_gt)),
        "nc_mean": float(nc1.mean()), "nc_med": float(np.median(nc1)),
        "nc2_mean": float(nc2.mean()), "nc2_med": float(np.median(nc2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--side", choices=["torch", "cpp", "paper"], default="torch")
    ap.add_argument("--cli", default=str(CPP / "build-cuda/bin/vggt-cli"))
    ap.add_argument("--gguf", default="")   # default per --arch
    ap.add_argument("--arch", choices=["pi3", "pi3x"], default="pi3")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--scenes", default="")   # comma filter, default all
    ap.add_argument("--conf-mask", action="store_true",
                    help="also drop the lowest-quintile-confidence predictions")
    ap.add_argument("--out", default=str(CPP / "benchmarks/results/"
                                         "eval_pi3_paper_eth3d.md"))
    args = ap.parse_args()
    args.out = Path(args.out)

    if not args.gguf:
        args.gguf = str(CPP / ("models/gguf/"
                               + ("pi3x-f16.gguf" if args.arch == "pi3x"
                                  else "pi3-f16.gguf")))
    if args.out.name == "eval_pi3_paper_eth3d.md":
        # keep per-arch results separate by default
        args.out = args.out.with_name(
            f"eval_pi3_paper_eth3d_{args.arch}_{args.side}.md")

    if args.side == "paper":
        print("pi3 paper Table 3 ETH3D reference:", PAPER_ETH3D)
        return

    data_root = Path(args.data_root)
    scenes = sorted(p.name for p in data_root.iterdir()
                    if (p / "scene_meta.json").exists())
    if args.scenes:
        scenes = args.scenes.split(",")
    print(f"pi3 paper ETH3D protocol: {len(scenes)} scenes, "
          f"stride {args.stride}, side={args.side}")

    all_rows, per_scene = [], {}
    for scene in scenes:
        frames = load_scene(data_root, scene)
        for f in frames:
            f["image_abs"] = data_root / scene / f["image"]
            f["depth_abs"] = data_root / scene / f["depth"]
        hh, ww = pick_resolution(frames[0]["h"], frames[0]["w"])
        n_kf = len(frames[::args.stride])
        print(f"  {scene}: {len(frames)} frames -> {n_kf} keyframes @ {hh}x{ww}")

        if args.side == "torch":
            pts, conf = torch_predictions(frames, hh, ww, args.stride,
                                          args.arch)
        else:
            pts, conf = cpp_predictions(frames, hh, ww, args.stride,
                                        args.cli, args.gguf)
        gt_pc_cam, gt_nm_cam, c2ws = gt_cloud_for_scene(frames[::args.stride], hh, ww)
        m = scene_metrics(pts, conf, gt_pc_cam, gt_nm_cam, c2ws, args.conf_mask)
        per_scene[scene] = m
        all_rows.append(m)
        print(f"    acc {m['acc_mean']:.4f}/{m['acc_med']:.4f}  "
              f"comp {m['comp_mean']:.4f}/{m['comp_med']:.4f}  "
              f"nc {m['nc_mean']:.4f}/{m['nc_med']:.4f}")

    avg = {k: float(np.mean([r[k] for r in all_rows]))
           for k in all_rows[0]}
    print("\n== scene-averaged ==")
    for k, v in avg.items():
        print(f"{k:12s} {v:.4f}")
    print("\npi3 paper reference:", PAPER_ETH3D)

    lines = [f"# pi3 paper ETH3D protocol (Table 3): stride {args.stride}, "
             f"width {MODEL_W}, Umeyama+ICP, side={args.side}, "
             f"{len(scenes)} scenes", "",
             "| metric | value | paper |", "|---|---|---|"]
    for k in ("acc_mean", "acc_med", "comp_mean", "comp_med",
              "nc_mean", "nc_med"):
        pk = k if k in PAPER_ETH3D else ("nc2_" + k[3:] if k.startswith("nc2") else None)
        pref = PAPER_ETH3D.get(k, "")
        lines.append(f"| {k} | {avg[k]:.4f} | {pref} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
