#!/usr/bin/env python3
"""vggt-1b (official facebook/VGGT-1B) end-to-end reconstruction comparison
on REAL ETH3D data — the pose_enc(9)+depth counterpart of
scripts/compare_reconstruction_pi3x.py (pi3/pi3x family).

Official PyTorch VGGT-1B vs the C++ ggml CLI across quantizations, same
window / inputs / metric code.  vggt-1b contract:
  * pose = 9 floats (t(3) | quat xyzw) per view — cpp .pose.bin (S,9);
  * depth = the model's dedicated depth head (z-depth) — cpp .depth.bin.
Alignment/protocol identical to the pi3-family script (avg_dis scale
alignment, view0-frame clouds via the model's own poses, same metrics).

Usage (PYTHONPATH=/tmp/torch_cuda_lib:<repo>):
  python3 scripts/compare_reconstruction_vggt.py \
      --data-root <...>/eth3d --metadata-dir <...>/metadata [--scene courtyard]
"""
import argparse
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CPP / "scripts"))

from eval_cpp_cli import (  # noqa: E402
    cloud_metrics, depth_metrics, load_wai_scene, load_wai_view,
    top_covisibility_window, unproject_depth,
)
from mapanything.utils.cropping import crop_resize_if_necessary  # noqa: E402
from mapanything.utils.geometry import (  # noqa: E402
    quaternion_to_rotation_matrix,
)
from compare_reconstruction_pi3x import (  # noqa: E402  shared plotters
    chart_cloud, chart_depth, chart_metrics, parse_official_md,
)

QUANTS = ["f16", "q8_0", "q5_K"]  # q6_K excluded: weakest tier for vggt-1b


@contextmanager
def tempfile_TemporaryDirectory():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        yield td


def pose9_to_se3(row):
    quat = torch.tensor(row[3:7], dtype=torch.float64).unsqueeze(0)
    R = quaternion_to_rotation_matrix(quat)[0].to(torch.float64)
    T = np.eye(4)
    T[:3, :3] = R.numpy()
    T[:3, 3] = row[:3]
    return T


def run_torch_vggt1b(imgs_f):
    import gc
    sys.path.insert(0, "/tmp/vggt-src")
    from vggt.models.vggt import VGGT
    m = VGGT().eval().requires_grad_(False)
    sd = torch.load(str(CPP / "models/pytorch/vggt1b/model.pt"),
                    map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model", sd.get("state_dict", sd)))
    m = m.cuda()
    with torch.inference_mode():
        r = m(torch.from_numpy(imgs_f).unsqueeze(0).cuda())
    depth = r["depth"].float().cpu().numpy().reshape(len(imgs_f),
                                                     *imgs_f.shape[2:])
    pose = r["pose_enc"].float().cpu().numpy().reshape(len(imgs_f), 9)
    # free the torch model BEFORE the cpp CLI allocates CUDA buffers
    del m, r, sd
    gc.collect()
    torch.cuda.empty_cache()
    return pose, depth


def run_cpp(cli, gguf, imgs_f, H, W):
    with tempfile_TemporaryDirectory() as td:
        frames_bin = Path(td) / "frames.bin"
        imgs_f.tofile(frames_bin)
        prefix = str(Path(td) / "out")
        proc = subprocess.run(
            [cli, "--model", gguf, "--bin", str(frames_bin),
             "--H", str(H), "--W", str(W), "--S", str(len(imgs_f)),
             "--out-prefix", prefix], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"CLI failed:\n{proc.stderr[-500:]}")
        pose = np.fromfile(f"{prefix}.pose.bin", dtype=np.float32
                           ).reshape(len(imgs_f), 9)
        depth = np.fromfile(f"{prefix}.depth.bin", dtype=np.float32
                            ).reshape(len(imgs_f), H, W)
    return pose, depth


def rot_trans_diff(pose_a, pose_b):
    out = []
    for a, b in zip(pose_a, pose_b):
        A, B = pose9_to_se3(a), pose9_to_se3(b)
        c = (np.trace(A[:3, :3] @ B[:3, :3].T) - 1.0) / 2.0
        rot = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        out.append((rot, float(np.linalg.norm(A[:3, 3] - B[:3, 3]))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--cli", default=str(CPP / "build-cuda/bin/vggt-cli"))
    ap.add_argument("--gguf-dir", default=str(CPP / "models/gguf"))
    ap.add_argument("--arch", default="vggt-1b",
                    help="gguf stem prefix + report label")
    ap.add_argument("--scene", default="")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--resolution", type=int, nargs=2, default=[518, 336])
    ap.add_argument("--out-dir", default=str(CPP / "benchmarks/charts/vggt-1b"))
    ap.add_argument("--results-dir",
                    default=str(CPP / "benchmarks/results/vggt-1b"))
    args = ap.parse_args()
    RW, RH = args.resolution
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = np.load(Path(args.metadata_dir) / "test" /
                     "eth3d_scene_list_test.npy", allow_pickle=True)
    scene = args.scene or str(scenes[0])
    scene_root = Path(args.data_root) / scene
    meta, cov = load_wai_scene(scene_root)
    frames = meta["frames"]
    window = top_covisibility_window(cov, min(args.views, len(frames)))
    print(f"scene={scene} window={window}")

    imgs, depths, Ks, valids, poses = [], [], [], [], []
    for idx in window:
        img, depth, K, pose = load_wai_view(scene_root, frames[idx])
        img, depth, K = crop_resize_if_necessary(
            image=img, resolution=(RW, RH), depthmap=depth,
            intrinsics=K, additional_quantities=None)
        imgs.append(np.asarray(img))
        depths.append(depth)
        Ks.append(K)
        valids.append(depth > 1e-3)
        poses.append(np.asarray(pose, dtype=np.float64))
    S = len(imgs)
    imgs_f = np.stack([np.transpose(im, (2, 0, 1)) for im in imgs]
                      ).astype(np.float32) / 255.0

    print("inference: torch VGGT-1B ...")
    pose_t, depth_t = run_torch_vggt1b(imgs_f)
    preds = {"torch": depth_t}
    poses_all = {"torch": pose_t}
    for q in QUANTS:
        gguf = Path(args.gguf_dir) / f"{args.arch}-{q}.gguf"
        if not gguf.exists():
            print(f"skip {q}: {gguf.name} missing")
            continue
        print(f"inference: cpp {q} ...")
        pose_c, depth_c = run_cpp(args.cli, str(gguf), imgs_f, RH, RW)
        preds[q] = depth_c
        poses_all[q] = pose_c

    # avg_dis scale alignment (repo protocol)
    def avg_dis_factor(pts_list, valids):
        return float(np.mean([
            np.linalg.norm(p.reshape(-1, 3)[v.reshape(-1)], axis=1).mean()
            for p, v in zip(pts_list, valids)]))

    gt_pts = [unproject_depth(depths[i], Ks[i]) for i in range(S)]
    gt_factor = avg_dis_factor(gt_pts, valids)
    aligned, factors = {}, {}
    for name, d in preds.items():
        pts = [unproject_depth(d[i], Ks[i]) for i in range(S)]
        f = avg_dis_factor(pts, valids)
        factors[name] = f
        aligned[name] = d * (gt_factor / f)

    chart_depth(imgs, depths, aligned, valids,
                out_dir / "recon_depth_comparison.png")

    def cloud_view0(depth, factor, pose_rows):
        out = []
        for i in range(S):
            T = (pose9_to_se3(pose_rows[i]) if pose_rows is not None
                 else np.linalg.inv(poses[0]) @ poses[i])
            pc = (unproject_depth(depth[i], Ks[i]) / factor).reshape(-1, 3)
            pw = pc @ T[:3, :3].T + T[:3, 3]
            out.append((pw, imgs[i].reshape(-1, 3) / 255.0,
                        valids[i].reshape(-1)))
        pts = np.concatenate([o[0] for o in out])
        rgb = np.concatenate([o[1] for o in out])
        msk = np.concatenate([o[2] for o in out])
        return pts[msk], rgb[msk]

    clouds = {"GT": [cloud_view0(depths, gt_factor, None)]}
    for name in preds:
        clouds[name] = [cloud_view0(aligned[name], factors[name],
                                    poses_all[name])]
    chart_cloud(clouds, f"{scene}, {S} views",
                out_dir / "recon_pointcloud_comparison.png")

    rows = {}
    for name, d in aligned.items():
        rr = []
        for i in range(S):
            dm = depth_metrics(depths[i] / gt_factor, d[i] / gt_factor,
                               valids[i])
            cm = cloud_metrics(gt_pts[i] / gt_factor,
                               unproject_depth(preds[name][i], Ks[i])
                               / factors[name], valids[i])
            rr.append({**dm, **cm})
        rows[name] = {k: float(np.nanmean([r[k] for r in rr]))
                      for k in rr[0]}
        if name != "torch":
            rd = rot_trans_diff(pose_t, poses_all[name])
            rows[name]["rot_deg"] = float(np.mean([r for r, _ in rd]))
            rows[name]["trans_m"] = float(np.mean([t for _, t in rd]))

    official = parse_official_md(Path(args.results_dir) /
                                 f"bench_official_eth3d_vggt.md")
    if official:
        chart_metrics(official, out_dir / "recon_metrics_comparison.png")

    keys = ["AbsRel", "SqRel", "RMSE", "RMSE-log", "d1", "accuracy",
            "completeness", "chamfer", "fscore", "rot_deg", "trans_m"]
    lines = [f"# {args.arch} reconstruction comparison — scene "
             f"`{scene}`, window {window}, {RW}x{RH}", "",
             "depth = the model's dedicated depth head (z-depth), "
             "scale-aligned to the GT avg_dis factor (repo protocol);",
             "pose diff = cpp vs torch (mean over views, quat->se3).", "",
             "| metric | " + " | ".join(rows.keys()) + " |",
             "|" + "--------|" * (len(rows) + 1)]
    for k in keys:
        lines.append(f"| {k} | " +
                     " | ".join(f"{rows[n].get(k, float('nan')):.6f}"
                                for n in rows) + " |")
    (out_dir / "recon_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("charts written:", *[p.name for p in sorted(
        out_dir.glob("recon_*"))], sep="\n  ")


if __name__ == "__main__":
    main()
