#!/usr/bin/env python3
"""pi3-family end-to-end reconstruction comparison on REAL ETH3D data —
the pi3/pi3x counterpart of scripts/compare_reconstruction.py (vggt-omega).

Official PyTorch Pi3/Pi3X vs the C++ ggml CLI across quantizations, same
window / inputs / metric code (--arch selects the family; both share the
16-float c2w + local_points contract).  Contract differences vs vggt-omega:
  * poses are 4x4 camera-to-world (row-major, metric) — not pose_enc(9);
  * depth = local_points[..., 2] (camera-frame z; official forward already
    multiplies local_points/points/poses by the metric scale);
  * global points are the model's world-frame cloud (pi3x world == view0).

Outputs (benchmarks/charts/pi3x/):
  recon_depth_comparison.png      rows = {input/GT, torch, f16, q8_0, q6_K}
  recon_pointcloud_comparison.png world-frame colored clouds (view0 frame)
  recon_metrics_comparison.png    official ETH3D protocol bars (130-set md)
  recon_comparison.md             numeric table for this window

Usage (PYTHONPATH=/tmp/torch_cuda_lib:<repo>):
  python3 scripts/compare_reconstruction_pi3x.py \
      --data-root <...>/eth3d --metadata-dir <...>/metadata [--scene courtyard]
"""
import argparse
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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

QUANTS = ["f16", "q8_0", "q6_K"]  # default K tier; pi3 overrides to q5_K


@contextmanager
def tempfile_TemporaryDirectory():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        yield td


def _quat_xyzw_to_mat(q):
    """xyzw scalar-last quaternion -> 3x3 rotation (row-major numpy)."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def run_torch_pi3x(imgs_f, arch="pi3x"):
    import gc
    if arch == "mapanything":
        # official facebook/map-anything: views list with ImageNet-normalized
        # imgs and the encoder-declared data_norm_type (asserted, not applied)
        sys.path.insert(0, str(CPP.parent))
        from mapanything.models.mapanything.model import MapAnything
        ckpt = CPP / "models/pytorch/mapanything"
        m = MapAnything.from_pretrained(str(ckpt))
        m = m.cuda().eval().requires_grad_(False)
        raw = torch.from_numpy(imgs_f).unsqueeze(0).cuda()   # (1,S,3,H,W)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device="cuda").view(1, 3, 1, 1)
        norm = (raw - mean) / std
        views = [{"img": norm[:, v], "data_norm_type": ["dinov2"]}
                 for v in range(imgs_f.shape[0])]
        with torch.inference_mode():
            res = m(views)
        S = imgs_f.shape[0]
        hw = imgs_f.shape[2:4]                     # imgs_f = (S,3,H,W) CHW
        depth = np.stack([                          # batch/shape tail dims;
            res[v]["depth_along_ray"].float().cpu().numpy().reshape(hw)
            for v in range(S)])                     # reshape to the known
        pts = np.stack([                            # input geometry instead
            res[v]["pts3d"].float().cpu().numpy().reshape(hw + (3,))
            for v in range(S)])
        pose = np.stack([
            np.vstack([np.concatenate([
                _quat_xyzw_to_mat(
                    res[v]["cam_quats"].float().cpu().numpy().reshape(4)),
                res[v]["cam_trans"].float().cpu().numpy().reshape(3, 1),
            ], axis=1), [0.0, 0.0, 0.0, 1.0]]).reshape(-1)   # full 4x4 c2w,
            for v in range(S)])                              # metric trans
        np.stack([depth, depth]).tofile(Path("_ma_torch_depth_diag.bin")
                                        ) if False else None
        depth.tofile(CPP / "benchmarks/charts" / "_ma_torch_depth.bin")
        del m, res
        gc.collect()
        torch.cuda.empty_cache()
        return pose, depth, pts
    sys.path.insert(0, str(CPP / "third_party" / "pi3-src"))
    if arch == "pi3x":
        from pi3.models.pi3x import Pi3X as Model
        ckpt = CPP / "models/pytorch/pi3x"
    else:
        from pi3.models.pi3 import Pi3 as Model
        ckpt = CPP / "models/pytorch/pi3"
    m = Model.from_pretrained(str(ckpt))
    m = m.cuda().eval().requires_grad_(False)
    with torch.inference_mode():
        r = m(torch.from_numpy(imgs_f).unsqueeze(0).cuda())
    lp = r["local_points"][0].float().cpu().numpy()          # (S,H,W,3) metric
    depth = lp[..., 2]                                        # camera z
    pose = r["camera_poses"][0].float().cpu().numpy().reshape(len(imgs_f), 16)
    pts = r["points"][0].float().cpu().numpy()                # world (metric)
    # free the torch model BEFORE the cpp CLI allocates its CUDA buffers
    # (pi3x fp32 weights + activations ~7 GB; the cpp graph needs ~1.5 GB)
    del m, r
    gc.collect()
    torch.cuda.empty_cache()
    return pose, depth, pts


def run_torch_dust3r(imgs_f):
    """Pair-wise M5: two forwards (A: v0,v1; B: v1,v0 swap); the closed-form
    assemble_dust3r (shared with bench_official_eth3d, no K / no GT)
    recovers view1's pose; view0 = identity.  Returns the common contract:
    (pose (S,16) row-major c2w, depth (S,H,W) = pts3d_cam z,
    pts (S,H,W,3) both pointmaps in view0's frame)."""
    import gc
    from bench_official_eth3d import assemble_dust3r   # same code both sides
    sys.path.insert(0, str(CPP / "third_party" / "dust3r-src"))
    import dust3r.utils.path_to_croco  # noqa: F401
    from dust3r.model import AsymmetricCroCo3DStereo
    m = AsymmetricCroCo3DStereo.from_pretrained(
        str(CPP / "models/pytorch/dust3r")).cuda().eval().requires_grad_(False)
    S, _, H, W = imgs_f.shape
    im = torch.from_numpy(imgs_f).unsqueeze(0).cuda() * 2.0 - 1.0
    shape = torch.tensor([[H, W]], device="cuda")
    with torch.inference_mode():
        rA1, rA2 = m(dict(img=im[:, 0], true_shape=shape, instance="a"),
                     dict(img=im[:, 1], true_shape=shape, instance="b"))
        rB1, _ = m(dict(img=im[:, 1], true_shape=shape, instance="b"),
                   dict(img=im[:, 0], true_shape=shape, instance="a"))
        preds, _ = assemble_dust3r(rA1["pts3d"], rA2["pts3d_in_other_view"],
                                   rB1["pts3d"], rA1["conf"], rB1["conf"])
    depth = np.stack([preds[v]["pts3d_cam"][0, ..., 2].float().cpu().numpy()
                      for v in range(S)])
    pts = np.stack([preds[v]["pts3d"][0].float().cpu().numpy()
                    for v in range(S)])
    pose = np.zeros((S, 16), dtype=np.float32)
    pose[0] = np.eye(4, dtype=np.float32).reshape(-1)
    P = np.eye(4, dtype=np.float32)
    P[:3, :3] = _quat_xyzw_to_mat(
        preds[1]["cam_quats"][0].float().cpu().numpy())
    P[:3, 3] = preds[1]["cam_trans"][0].float().cpu().numpy()
    pose[1] = P.reshape(-1)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return pose, depth, pts


def run_cpp_dust3r(cli, gguf, imgs_f, H, W):
    """Two CLI runs (A: v0,v1; B: v1,v0 swap) + the shared closed-form
    assemble — identical formula to the torch side."""
    import torch
    from bench_official_eth3d import assemble_dust3r
    S = len(imgs_f)
    with tempfile_TemporaryDirectory() as td:
        raw = {}
        for tag, fr in (("a", imgs_f), ("b", imgs_f[::-1].copy())):
            frames_bin = Path(td) / f"frames_{tag}.bin"
            fr.tofile(frames_bin)
            prefix = str(Path(td) / f"out_{tag}")
            proc = subprocess.run(
                [cli, "--model", gguf, "--bin", str(frames_bin),
                 "--H", str(H), "--W", str(W), "--S", "2",
                 "--out-prefix", prefix], capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(f"CLI failed:\n{proc.stderr[-500:]}")
            lp = np.fromfile(f"{prefix}.local_points.bin",
                             dtype=np.float32).reshape(2, H, W, 3)
            cf = np.fromfile(f"{prefix}.conf.bin",
                             dtype=np.float32).reshape(2, H, W, 1)
            raw[tag] = (lp, cf)
    t0 = lambda a: torch.from_numpy(a).cuda().unsqueeze(0)  # noqa: E731
    with torch.inference_mode():
        preds, _ = assemble_dust3r(
            t0(raw["a"][0][0]), t0(raw["a"][0][1]), t0(raw["b"][0][0]),
            t0(raw["a"][1][0]), t0(raw["b"][1][0]))
    depth = np.stack([preds[v]["pts3d_cam"][0, ..., 2].float().cpu().numpy()
                      for v in range(S)])
    pts = np.stack([preds[v]["pts3d"][0].float().cpu().numpy()
                    for v in range(S)])
    pose = np.zeros((S, 16), dtype=np.float32)
    pose[0] = np.eye(4, dtype=np.float32).reshape(-1)
    P = np.eye(4, dtype=np.float32)
    P[:3, :3] = _quat_xyzw_to_mat(
        preds[1]["cam_quats"][0].float().cpu().numpy())
    P[:3, 3] = preds[1]["cam_trans"][0].float().cpu().numpy()
    pose[1] = P.reshape(-1)
    return pose, depth, pts


def run_cpp_pi3x(cli, gguf, imgs_f, H, W):
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
                           ).reshape(len(imgs_f), 16)
        lp = np.fromfile(f"{prefix}.local_points.bin", dtype=np.float32
                         ).reshape(len(imgs_f), H, W, 3)
        pts = np.fromfile(f"{prefix}.points.bin", dtype=np.float32
                          ).reshape(len(imgs_f), H, W, 3)
        # depth: mapanything's official "depth" is the dense head's
        # depth-along-ray (.depth.bin), NOT the local-point z (pi3/pi3x have
        # no depth head, so their .depth.bin IS the local z — same file, per-
        # family semantics, both emitted by the CLI)
        db = Path(f"{prefix}.depth.bin")
        if db.exists() and db.stat().st_size == len(imgs_f) * H * W * 4:
            depth = np.fromfile(db, dtype=np.float32).reshape(len(imgs_f), H, W)
        else:
            depth = lp[..., 2]
    return pose, depth, pts


def rot_trans_diff(pose_a, pose_b):
    """pose rows are flat row-major 4x4 c2w; returns (rot_deg, trans_l2)."""
    out = []
    for a, b in zip(pose_a, pose_b):
        A, B = a.reshape(4, 4), b.reshape(4, 4)
        c = (np.trace(A[:3, :3] @ B[:3, :3].T) - 1.0) / 2.0
        rot = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        out.append((rot, float(np.linalg.norm(A[:3, 3] - B[:3, 3]))))
    return out


def to_view0(pts_world, pose16):
    """world points -> view0 camera frame (pi3x world == view0, so this is
    near-identity; kept explicit for robustness)."""
    P = pose16.reshape(4, 4)
    R0, t0 = P[:3, :3], P[:3, 3]
    return (pts_world - t0) @ R0


def chart_depth(imgs, gt, preds, valids, out):
    names = list(preds.keys())
    S = len(imgs)
    fig, axes = plt.subplots(1 + len(names), 4,
                             figsize=(3.2 * 4, 2.5 * (1 + len(names))),
                             squeeze=False)
    vmin, vmax = np.percentile(gt[0][valids[0]], [2, 98])
    for j in range(S):
        axes[0][j].imshow(imgs[j] / 255.0)
        axes[0][j].set_title(f"input v{j}", fontsize=10)
        axes[0][j].axis("off")
        ax = axes[0][2 + j]
        im = ax.imshow(np.where(valids[j], gt[j], np.nan), cmap="magma",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"GT depth v{j}", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for r, name in enumerate(names, start=1):
        d = preds[name]
        for j in range(S):
            ax = axes[r][j]
            ax.imshow(d[j], cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name} depth v{j}", fontsize=10)
            ax.axis("off")
            err = np.abs(d[j] - gt[j]) / np.maximum(gt[j], 1e-6)
            ax = axes[r][2 + j]
            im = ax.imshow(np.where(valids[j], np.clip(err, 0, 0.25), np.nan),
                           cmap="turbo", vmin=0, vmax=0.25)
            ax.set_title(f"{name} |err| v{j}", fontsize=10)
            ax.axis("off")
            if r == 1:
                fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _cloud_panel(ax, pts, rgb, lims, title):
    lo, hi = lims
    ax.scatter(pts[:, 0], pts[:, 1], s=0.9, linewidths=0,
               c=np.clip(rgb, 0, 1))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def chart_cloud(clouds, title_sub, out):
    names = list(clouds.keys())
    S = len(clouds[names[0]])
    allpts = np.concatenate([p for n in names for p, _ in clouds[n]])
    zmax = np.percentile(allpts[:, 2], 98)
    lo = np.percentile(allpts[:, :2], 1, axis=0) - 1.0
    hi = np.percentile(allpts[:, :2], 99, axis=0) + 1.0
    lims = (lo, hi)
    fig, axes = plt.subplots(2, len(names),
                             figsize=(3.4 * len(names), 7.6), squeeze=False)
    for k, name in enumerate(names):
        for s, (pts, rgb) in enumerate(clouds[name]):
            keep = pts[:, 2] <= zmax
            label = name if s == 0 else f"{name} (v{s})"
            _cloud_panel(axes[0][k], pts[keep][:, [0, 1]], rgb[keep],
                         lims, label if s == 0 else "")
            ax = axes[1][k]
            ax.scatter(pts[keep][:, 0], pts[keep][:, 2], s=0.9,
                       linewidths=0, c=np.clip(rgb[keep], 0, 1))
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(0, zmax)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.axis("off")
            if s == 0:
                ax.set_title(f"{name}\ntop view (x, z↓)", fontsize=10)
    fig.suptitle(f"ETH3D reconstruction — view0-frame colored point "
                 f"clouds ({title_sub})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_metrics(official, out):
    """official: {metric: {src: value}} parsed from the 130-set pi3x bench."""
    groups = [
        ("AbsRel (log)", ["z_depth_abs_rel", "pointmaps_abs_rel"], "log"),
        ("inlier@1.03 (x100)", ["z_depth_inlier_thres_103",
                                "pointmaps_inlier_thres_103"], "linear"),
        ("Pose AUC@5 / ATE", ["pose_auc_5", "pose_ate_rmse"], "log"),
        ("rotation / center err", ["rot_err_deg", "center_err"], "log"),
    ]
    srcs = list(dict.fromkeys(s for m in official.values() for s in m))
    colors = {"torch": "#1f77b4", "cpp": "#ff7f0e"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (title, keys, yscale) in zip(axes, groups):
        x = 0.0
        for ki, key in enumerate(keys):
            width = 0.8 / max(len(srcs), 1)
            for si, src in enumerate(srcs):
                v = official.get(key, {}).get(src)
                if v is None:
                    continue
                ax.bar(x + si * width, max(v, 1e-6), width, label=src
                       if ki == 0 else None, color=colors.get(src, "#777"))
                ax.annotate(f"{v:.4g}", (x + si * width, max(v, 1e-6)),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=6, rotation=90)
            x += 1.0
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)
        ax.set_yscale(yscale)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Official MapAnything ETH3D protocol (FULL 130 sets): "
                 "torch Pi3X f32 vs cpp pi3x-f16", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def parse_official_md(path):
    out = {}
    header = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("|") and "---" not in line:
            cells = [p.strip() for p in line.strip("|").split("|")]
            if cells[0] == "metric":
                header = cells[1:]
            elif header and len(cells) == len(header) + 1:
                try:
                    vals = [float(v) for v in cells[1:]]
                except ValueError:
                    continue
                for src, v in zip(header, vals):
                    out.setdefault(cells[0], {})[src] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--cli", default=str(CPP / "build-cuda/bin/vggt-cli"))
    ap.add_argument("--gguf-dir", default=str(CPP / "models/gguf"))
    ap.add_argument("--arch", default="pi3x",
                    choices=["pi3", "pi3x", "mapanything", "dust3r"])
    ap.add_argument("--scene", default="")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--resolution", type=int, nargs=2, default=[518, 336])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--results-dir", default="")
    args = ap.parse_args()
    if args.arch == "dust3r" and args.resolution == [518, 336]:
        args.resolution = [512, 336]   # patch 16 (pair-wise M5)
    if not args.out_dir:
        args.out_dir = str(CPP / "benchmarks/charts" / args.arch)
    if not args.results_dir:
        args.results_dir = str(CPP / "benchmarks/results" / args.arch)
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

    print(f"inference: torch {args.arch} ...")
    if args.arch == "dust3r":
        pose_t, depth_t, pts_t = run_torch_dust3r(imgs_f)
    else:
        pose_t, depth_t, pts_t = run_torch_pi3x(imgs_f, args.arch)
    preds = {"torch": depth_t}
    poses_all = {"torch": pose_t}
    pts_all = {"torch": pts_t}
    # ONE measured K tier per model (2026-10-01): pi3 ships q5_K, the rest q6_K
    k_tier = "q5_K" if args.arch == "pi3" else "q6_K"
    for q in ["f16", "q8_0", k_tier]:
        gguf = Path(args.gguf_dir) / f"{args.arch}-{q}.gguf"
        if not gguf.exists():
            print(f"skip {q}: {gguf.name} missing")
            continue
        print(f"inference: cpp {q} ...")
        if args.arch == "dust3r":
            pose_c, depth_c, pts_c = run_cpp_dust3r(args.cli, str(gguf),
                                                    imgs_f, RH, RW)
        else:
            pose_c, depth_c, pts_c = run_cpp_pi3x(args.cli, str(gguf),
                                                  imgs_f, RH, RW)
        preds[q] = depth_c
        poses_all[q] = pose_c
        pts_all[q] = pts_c

    # ---- avg_dis scale alignment (the established repo protocol, same as
    # compare_reconstruction.py): align each prediction set to GT depth so
    # the comparison is scale-fair regardless of the GT depth unit ----
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

    def pose_to_se3(pose16):
        P = np.asarray(pose16, dtype=np.float64).reshape(4, 4)
        return P

    def cloud_view0(depth, factor, pose16):
        """aligned-depth unprojection placed in the view0 frame via the
        model's own per-view poses (GT uses the WAI c2w poses)."""
        out = []
        for i in range(S):
            T = pose_to_se3(pose16[i]) if pose16 is not None else \
                np.linalg.inv(poses[0]) @ poses[i]
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

    # window metrics per source (identical functions as the eval matrix)
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
                                 f"bench_official_eth3d_{args.arch}.md")
    if official:
        chart_metrics(official, out_dir / "recon_metrics_comparison.png")

    keys = ["AbsRel", "SqRel", "RMSE", "RMSE-log", "d1", "accuracy",
            "completeness", "chamfer", "fscore", "rot_deg", "trans_m"]
    lines = [f"# {args.arch} reconstruction comparison — scene "
             f"`{scene}`, window {window}, {RW}x{RH}", "",
             "depth = local_points[..., 2] (camera z, metric), scale-aligned",
             "to the GT avg_dis factor (repo protocol); pose diff = cpp vs",
             "torch (mean over views, metric frames).", "",
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
