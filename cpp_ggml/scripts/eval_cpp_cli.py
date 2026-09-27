#!/usr/bin/env python3
"""End-to-end evaluation of the C++ CLI on the official ETH3D protocol.

Single metric implementation shared by both runtimes (PyTorch and the C++
CLI feed the same functions), which is what guarantees parity of the metric
definitions with the Python version.

Bridge from CLI outputs to the official MapAnything ETH3D protocol
(reproduction.md: AUC / ATE / Point AbsRel / Depth AbsRel, computed on the
z-depth + camera-frame pointmap predicted from GT intrinsics):
  cam_quats/cam_trans <- pose_enc (trans3+quat4+fov2)
  depth               <- depth.bin
  pts3d_cam           <- unproject(depth, GT intrinsics)
  view0-frame pts3d   <- geotrf(inv(pred_view0_pose), pts3d_cam) (official)
  ray directions      <- GT intrinsics (ray dir error is pose-driven)

View sampling: deterministic top-covisibility window per scene (the official
loader uses a random walk; any connected window is valid for runtime parity
since both runtimes see identical inputs).

Usage (PYTHONPATH=/tmp/torch_cuda_lib:<repo>):
  python3 scripts/eval_cpp_cli.py \
      --data-root <...>/eth3d --metadata-dir <...>/metadata \
      --cli build-cuda/bin/vggt-cli \
      --gguf models/gguf/vggt-omega-1b-512-f16.gguf \
      --num-views 2 --max-sets 13 --out benchmarks/results/eval_eth3d_cpp.md
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import PIL.Image
import torch

REPO = Path(__file__).resolve().parent.parent.parent
CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mapanything.utils.cropping import crop_resize_if_necessary  # noqa: E402
from mapanything.utils.geometry import (  # noqa: E402
    geotrf,
    inv,
    quaternion_to_rotation_matrix,
)
from mapanything.utils.metrics import (  # noqa: E402
    calculate_auc_np,
    evaluate_ate,
    se3_to_relative_pose_error,
)


# ------------------------------------------------------------------ WAI I/O
def load_wai_scene(scene_root):
    meta = json.load(open(Path(scene_root) / "scene_meta.json"))
    frames = meta["frames"]
    cov_path = Path(scene_root) / "covisibility" / "v0" / \
        f"pairwise_covisibility--{len(frames)}x{len(frames)}.npy"
    cov = np.load(cov_path) if cov_path.exists() else np.eye(len(frames))
    return meta, cov


def top_covisibility_window(cov, n):
    """Deterministic window: the view with the highest total covisibility
    plus its (n-1) most covisible partners."""
    totals = cov.sum(axis=1)
    anchor = int(np.argmax(totals))
    scores = cov[anchor].copy()
    scores[anchor] = -1
    partners = np.argsort(-scores)[: n - 1]
    return [anchor] + list(partners)


def load_wai_view(scene_root, frame):
    img = np.asarray(PIL.Image.open(
        Path(scene_root) / frame["image"]).convert("RGB"))
    depth = np.asarray(iio.imread(Path(scene_root) / frame["depth"]),
                       dtype=np.float32)
    K = np.array([[frame["fl_x"], 0, frame["cx"]],
                  [0, frame["fl_y"], frame["cy"]],
                  [0, 0, 1]], dtype=np.float32)
    pose = np.array(frame["transform_matrix"], dtype=np.float32)  # c2w
    return img, depth, K, pose


def unproject_depth(depth, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    yy, xx = np.mgrid[0:H, 0:W]
    return np.stack([(xx - cx) / fx * depth, (yy - cy) / fy * depth, depth],
                    axis=-1)


# --------------------------------------------------------------- metrics
def depth_metrics(gt, pred, mask):
    m = mask & (gt > 1e-3)
    if m.sum() == 0:
        return {k: np.nan for k in ("abs_rel", "sq_rel", "rmse", "rmse_log",
                                    "d1", "d2", "d3")}
    g, p = gt[m], pred[m]
    thresh = np.maximum(g / p, p / g)
    return {
        "AbsRel": float(np.mean(np.abs(p - g) / g)),
        "SqRel": float(np.mean((p - g) ** 2 / g)),
        "RMSE": float(np.sqrt(np.mean((p - g) ** 2))),
        "RMSE-log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25 ** 2)),
        "d3": float(np.mean(thresh < 1.25 ** 3)),
    }


def cloud_metrics(gt_pts, pred_pts, mask, fscore_thresh=0.05):
    """Camera-frame Chamfer family (extension column; one shared
    implementation feeds both runtimes). Keys match the aggregation list."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {k: np.nan for k in
                ("chamfer", "accuracy", "completeness", "fscore")}
    g = gt_pts.reshape(-1, 3)[mask.reshape(-1)]
    p = pred_pts.reshape(-1, 3)
    if len(g) == 0 or len(p) == 0:
        return {k: np.nan for k in
                ("chamfer", "accuracy", "completeness", "fscore")}
    rng = np.random.default_rng(0)
    gs = g[rng.choice(len(g), min(len(g), 65536), False)]
    ps = p[rng.choice(len(p), min(len(p), 65536), False)]
    d_gp, _ = cKDTree(ps).query(gs, k=1)
    d_pg, _ = cKDTree(gs).query(ps, k=1)
    acc, comp = float(np.mean(d_gp)), float(np.mean(d_pg))
    return {"accuracy": acc, "completeness": comp,
            "chamfer": 0.5 * (acc + comp),
            "fscore": float(np.mean(d_gp < fscore_thresh) * 0.5 +
                            np.mean(d_pg < fscore_thresh) * 0.5)}


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--num-views", type=int, default=2)
    ap.add_argument("--max-sets", type=int, default=13)
    ap.add_argument("--resolution", type=int, nargs=2, default=[512, 336],
                    help="W H (official ETH3D protocol: 512_1_52_ar = 512x336)")
    ap.add_argument("--out", default="benchmarks/results/eval_eth3d.md")
    ap.add_argument("--pred-source", choices=["cpp", "torch"], default="cpp",
                    help="cpp = vggt-cli bins; torch = official model (same "
                         "protocol; adjudicates cpp bugs vs protocol gaps)")
    ap.add_argument("--pt", default="vggt-omega/vggt_omega_1b_512.pt",
                    help="torch-mode checkpoint path under models/pytorch/")
    args = ap.parse_args()
    RW, RH = args.resolution

    scenes = np.load(Path(args.metadata_dir) / "test" /
                     "eth3d_scene_list_test.npy", allow_pickle=True)
    print(f"ETH3D scenes: {len(scenes)}")

    keys = ["AbsRel", "SqRel", "RMSE", "RMSE-log", "d1", "d2", "d3",
            "rot_err_deg", "center_err", "ATE",
            "AUC@1", "AUC@5", "AUC@15", "AUC@30",
            "chamfer", "accuracy", "completeness", "fscore",
            "fov_err_deg", "fov_x_err_deg", "fov_y_err_deg",
            "fx_err", "fy_err"]
    agg = {k: [] for k in keys}
    n_done = 0

    for scene in scenes:
        scene_root = Path(args.data_root) / scene
        meta, cov = load_wai_scene(scene_root)
        frames = meta["frames"]
        n = min(args.num_views, len(frames))
        window = top_covisibility_window(cov, n)
        imgs, depths, Ks, poses, valids = [], [], [], [], []
        ok = True
        for idx in window:
            fr = frames[idx]
            img, depth, K, pose = load_wai_view(scene_root, fr)
            img, depth, K = crop_resize_if_necessary(
                image=img, resolution=(RW, RH), depthmap=depth,
                intrinsics=K, additional_quantities=None)
            img = np.asarray(img)  # crop returns a PIL image
            if depth.shape != (RH, RW) or img.shape[:2] != (RH, RW):
                ok = False
                break
            imgs.append(img)
            depths.append(depth)
            Ks.append(K)
            poses.append(pose)
            valids.append(depth > 1e-3)
        if not ok or len(imgs) < 1:
            print(f"{scene}: skip (resize mismatch)")
            continue

        S = len(imgs)
        imgs_f = np.stack([np.transpose(im, (2, 0, 1)) for im in imgs]
                          ).astype(np.float32) / 255.0
        if args.pred_source == "torch":
            # adjudication path: the official model on the SAME inputs, so
            # any metric gap vs the CLI row is a cpp bug, not a protocol one
            sys.path.insert(0, str(CPP / "third_party" / "vggt-omega-src"))
            from vggt_omega.models import VGGTOmega
            global _TORCH_STATE
            try:
                _TORCH_STATE[0]
            except NameError:
                _TORCH_MODEL = VGGTOmega().eval()
                sd = torch.load(CPP / "models" / "pytorch" / args.pt,
                                map_location="cpu", weights_only=False)
                state = sd.get("model", sd.get("state_dict", sd))
                # default ctor builds no text_alignment_head (the C++ graph
                # runs the enable_alignment=False path); drop those ckpt
                # entries so the 256_text checkpoint loads strictly
                model_keys = set(_TORCH_MODEL.state_dict().keys())
                _TORCH_MODEL.load_state_dict(
                    {k: v for k, v in state.items() if k in model_keys})
                _TORCH_MODEL = _TORCH_MODEL.cuda()
                _TORCH_STATE = [args.pt]
            if _TORCH_STATE[0] != args.pt:
                raise SystemExit("--pt changed mid-run; run one ckpt per process")
            with torch.inference_mode():
                preds = _TORCH_MODEL(
                    torch.from_numpy(imgs_f).unsqueeze(0).cuda())
            pose = preds["pose_enc"].float().cpu().numpy().reshape(S, 9)
            depth = preds["depth"].float().cpu().numpy().reshape(S, RH, RW)
        else:
            with tempfile.TemporaryDirectory() as td:
                frames_bin = Path(td) / "frames.bin"
                imgs_f.tofile(frames_bin)
                prefix = str(Path(td) / "out")
                proc = subprocess.run(
                    [args.cli, "--model", args.gguf, "--bin", str(frames_bin),
                     "--H", str(RH), "--W", str(RW), "--S", str(S),
                     "--out-prefix", prefix],
                    capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"{scene}: CLI failed\n{proc.stderr[-500:]}")
                    continue
                pose = np.fromfile(f"{prefix}.pose.bin",
                                   dtype=np.float32).reshape(S, 9)
                depth = np.fromfile(f"{prefix}.depth.bin",
                                    dtype=np.float32).reshape(S, RH, RW)

        set_m = {k: [] for k in keys if k not in
                 ("rot_err_deg", "center_err", "ATE", "AUC@1", "AUC@5",
                  "AUC@15", "AUC@30")}
        gt_se3, pr_se3 = [], []
        # Official MapAnything protocol (benchmark.py get_all_info): pr and gt
        # quantities are EACH normalized by their own avg_dis factor
        # (normalize_multiple_pointclouds, norm_mode="avg_dis") before the
        # metric computation, so the metric scale cancels out.
        pr_pts_list, gt_pts_list = [], []
        for i in range(S):
            pr_pts_list.append(unproject_depth(depth[i], Ks[i]).reshape(-1, 3))
            gt_pts_list.append(unproject_depth(depths[i], Ks[i]).reshape(-1, 3))
        pr_factor = float(np.mean([np.linalg.norm(
            p[v.reshape(-1)], axis=1).mean()
            for p, v in zip(pr_pts_list, [vv.reshape(-1) for vv in valids])]))
        gt_factor = float(np.mean([np.linalg.norm(
            p[v.reshape(-1)], axis=1).mean()
            for p, v in zip(gt_pts_list, [vv.reshape(-1) for vv in valids])]))
        for i in range(S):
            # depth metrics on avg_dis-normalized z-depths (official scale
            # convention: quantities are divided by their own avg factor)
            dm = depth_metrics(depths[i] / gt_factor, depth[i] / pr_factor,
                               valids[i])
            set_m.update(dm)
            # normalized camera-frame pointmaps (z-depth / own avg factor;
            # x/y scale with the same factor since unprojection is linear)
            gt_pts_cam = unproject_depth(depths[i] / gt_factor, Ks[i])
            pr_pts_cam = unproject_depth(depth[i] / pr_factor, Ks[i])
            cm = cloud_metrics(gt_pts_cam, pr_pts_cam, valids[i])
            for k, v in cm.items():
                set_m[k] = v
            # fov mapping (official pose_enc): pose[7] = fov_h (VERTICAL,
            # computed with H/fy), pose[8] = fov_w (HORIZONTAL, with W/fx)
            fov_vert, fov_hor = pose[i, 7], pose[i, 8]
            fx_gt, fy_gt = Ks[i][0, 0], Ks[i][1, 1]
            fov_x_gt = 2 * np.arctan(RW / (2 * fx_gt))
            fov_y_gt = 2 * np.arctan(RH / (2 * fy_gt))
            set_m["fov_x_err_deg"] = abs(
                float(np.degrees(fov_hor - fov_x_gt)))
            set_m["fov_y_err_deg"] = abs(
                float(np.degrees(fov_vert - fov_y_gt)))
            set_m["fov_err_deg"] = 0.5 * (set_m["fov_x_err_deg"] +
                                          set_m["fov_y_err_deg"])
            fx_pr = (RW / 2.0) / np.tan(fov_hor / 2.0)
            fy_pr = (RH / 2.0) / np.tan(fov_vert / 2.0)
            set_m["fx_err"] = abs(fx_pr - fx_gt) / fx_gt
            set_m["fy_err"] = abs(fy_pr - fy_gt) / fy_gt
            quat = torch.tensor(pose[i, 3:7], dtype=torch.float64).unsqueeze(0)
            R = quaternion_to_rotation_matrix(quat)[0].to(torch.float64)
            gt_se3.append(torch.tensor(np.asarray(poses[i], dtype=np.float64)))
            T = torch.eye(4, dtype=torch.float64)
            T[:3, :3] = R
            T[:3, 3] = torch.tensor(pose[i, :3], dtype=torch.float64)
            pr_se3.append(T)

        # poses are compared in the view-0 frame (official get_all_info
        # normalizes both gt and pred w.r.t. camera of view 0); the model
        # already predicts view0-relative poses, the GT poses are absolute
        gt_se3 = [torch.tensor(np.linalg.inv(
            gt_se3[0].numpy()) @ g.numpy()) for g in gt_se3]

        ate = evaluate_ate(gt_traj=gt_se3, est_traj=pr_se3)
        pr_stack, gt_stack = torch.stack(pr_se3), torch.stack(gt_se3)
        rel_r, rel_t = se3_to_relative_pose_error(pr_stack, gt_stack, S)
        r_err, t_err = rel_r.numpy(), rel_t.numpy()
        aucs = {f"AUC@{t}": float(calculate_auc_np(r_err, t_err,
                                                   max_threshold=t)[0])
                for t in (1, 5, 15, 30)}
        row = dict(set_m)
        row["rot_err_deg"] = float(np.mean(r_err))
        row["center_err"] = float(np.mean(np.linalg.norm(
            pr_stack[:, :3, 3].numpy() - gt_stack[:, :3, 3].numpy(), axis=1)))
        row["ATE"] = float(np.asarray(ate).ravel()[0])
        row.update(aucs)
        for k in keys:
            if k in row:
                agg[k].append(row[k])
        n_done += 1
        print(f"{scene}: done ({n_done} sets)")

    lines = [
        f"# ETH3D eval — {Path(args.gguf).name}",
        f"windows: {n_done} · views/window: {args.num_views} · "
        f"res: {RW}x{RH}",
        "",
        "| metric | value |",
        "|------|-----|",
    ]
    for k in keys:
        if agg[k]:
            lines.append(f"| {k} | {np.nanmean(agg[k]):.6f} |")
    md = "\n".join(lines) + "\n"
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
