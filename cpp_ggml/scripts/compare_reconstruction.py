#!/usr/bin/env python3
"""End-to-end reconstruction comparison on REAL ETH3D data:
official PyTorch checkpoint vs the C++ ggml CLI across QUANTIZATIONS,
same window / inputs / metric code.

  recon_depth_comparison.png      rows = {input/GT, torch, f16, q8_0, q6_K};
                                  cols = {depth v0, depth v1, err v0, err v1}
  recon_pointcloud_comparison.png official-style world-frame colored clouds
                                  (GT / torch / f16 / q8_0 / q6_K), front+top
  recon_metrics_comparison.png    OFFICIAL 8-metric protocol bars from
                                  scripts/bench_official_eth3d.py outputs
  recon_comparison.md             this-window numeric table

Usage (PYTHONPATH=/tmp/torch_cuda_lib:<repo>):
  python3 scripts/compare_reconstruction.py \
      --data-root <...>/eth3d --metadata-dir <...>/metadata [--scene courtyard]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CPP / "third_party" / "vggt-omega-src"))
sys.path.insert(0, str(CPP / "scripts"))

from eval_cpp_cli import (  # noqa: E402
    cloud_metrics, depth_metrics, load_wai_scene, load_wai_view,
    top_covisibility_window, unproject_depth,
)
from mapanything.utils.cropping import crop_resize_if_necessary  # noqa: E402
from mapanything.utils.geometry import quaternion_to_rotation_matrix  # noqa: E402

QUANTS = ["f16", "q8_0", "q6_K"]
QUANT_COLORS = {"torch": "#1f77b4", "f16": "#ff7f0e", "q8_0": "#2ca02c",
                "q6_K": "#9467bd"}


def avg_dis_factor(pts_list, valids):
    return float(np.mean([
        np.linalg.norm(p.reshape(-1, 3)[v.reshape(-1)], axis=1).mean()
        for p, v in zip(pts_list, valids)]))


def pose_to_se3(pose_row):
    quat = torch.tensor(pose_row[3:7], dtype=torch.float64).unsqueeze(0)
    R = quaternion_to_rotation_matrix(quat)[0].to(torch.float64)
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor(pose_row[:3], dtype=torch.float64)
    return T


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


from contextlib import contextmanager  # noqa: E402


@contextmanager
def tempfile_TemporaryDirectory():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        yield td


def run_torch(pt, imgs_f, H, W):
    from vggt_omega.models import VGGTOmega
    m = VGGTOmega().eval()
    sd = torch.load(CPP / "models" / "pytorch" / pt, map_location="cpu",
                    weights_only=False)
    m.load_state_dict(sd.get("model", sd.get("state_dict", sd)))
    m = m.cuda()
    with torch.inference_mode():
        preds = m(torch.from_numpy(imgs_f).unsqueeze(0).cuda())
    return (preds["pose_enc"].float().cpu().numpy().reshape(len(imgs_f), 9),
            preds["depth"].float().cpu().numpy().reshape(len(imgs_f), H, W))


def parse_official_md(path):
    """Parse an official-protocol md table -> {metric: {src: value}}."""
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


def chart_depth(imgs, gt, preds, valids, out):
    """preds: OrderedDict name -> depth (S,H,W), scale-aligned."""
    names = list(preds.keys())
    S = len(imgs)
    fig, axes = plt.subplots(1 + len(names), 4,
                             figsize=(3.2 * 4, 2.5 * (1 + len(names))),
                             squeeze=False)
    vmin, vmax = np.percentile(gt[0][valids[0]], [2, 98])
    for j in range(S):
        axes[0][j].imshow(imgs[j] / 255.0)
        axes[0][j].set_title(f"input v{j}", fontsize=10)
        axes[0 + 0][j].axis("off")
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
    """Official-style world-frame clouds. clouds: OrderedDict name ->
    list of (N,3) view0-frame arrays + matching rgb list."""
    names = list(clouds.keys())
    S = len(clouds[names[0]])
    # depth-trim + shared limits from ALL clouds (kills divergent outliers)
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
            # top view: x vs depth (z up), same trim
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
    fig.suptitle(f"ETH3D reconstruction — official world-frame colored point "
                 f"clouds ({title_sub})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _src_color(src):
    """Stable color per source label (quantization / backend aware)."""
    if "torch" in src:
        return "#1f77b4"
    if "q8_0" in src:
        return "#2ca02c"
    if "q6_K" in src:
        return "#d62728"
    if "cpu" in src:
        return "#9467bd"
    if "vulkan" in src:
        return "#17becf"
    return "#ff7f0e"  # f16 cuda (default accent)


def chart_metrics(official, out):
    """official: {metric: {src: value}} from bench_official_eth3d md files."""
    groups = [
        ("AbsRel (log)", ["z_depth_abs_rel", "pointmaps_abs_rel"], "log",
         False),
        ("inlier@1.03 (x100)", ["z_depth_inlier_thres_103",
                                "pointmaps_inlier_thres_103"], "linear",
         True),
        ("Pose AUC@5 (x100)", ["pose_auc_5"], "linear", False),
        ("camera error (deg / norm ATE, log)", ["ray_dirs_err_deg",
                                                "pose_ate_rmse"], "log",
         False),
    ]
    srcs = list(dict.fromkeys(s for m in official.values() for s in m))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, (title, keys, yscale, pct) in zip(axes, groups):
        x, width = 0, 0.8 / len(srcs)
        for si, src in enumerate(srcs):
            for ki, k in enumerate(keys):
                if k not in official or src not in official[k]:
                    continue
                v = official[k][src]
                if pct:
                    v = v * 100
                ax.bar(x + si * width, max(v, 1e-6), width,
                       label=src if ki == 0 else None,
                       color=_src_color(src))
                ax.annotate(f"{v:.4g}" if v < 2 else f"{v:.3g}",
                            (x + si * width, max(v, 1e-6)),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=6, rotation=90)
        x += 1.0
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)
        ax.set_yscale(yscale)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Official MapAnything ETH3D protocol (130 sets, random-walk "
                 "sampling, seed 777): torch vs cpp — quantizations & backends",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--cli", default=str(CPP / "build-cuda/bin/vggt-cli"))
    ap.add_argument("--gguf-dir", default=str(CPP / "models/gguf"))
    ap.add_argument("--pt", default="vggt-omega/vggt_omega_1b_512.pt")
    ap.add_argument("--scene", default="")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--resolution", type=int, nargs=2, default=[512, 336])
    ap.add_argument("--out-dir", default=str(CPP / "benchmarks/charts"))
    ap.add_argument("--results-dir", default=str(CPP / "benchmarks/results"))
    ap.add_argument("--official-md", nargs="*", default=[
        "bench_official_eth3d.md:torch:cpp f16 cuda",
        "bench_official_eth3d_q8_0.md:cpp:cpp q8_0 cuda",
        "bench_official_eth3d_q6_K.md:cpp:cpp q6_K cuda",
        "bench_official_eth3d_cpu.md:cpp:cpp f16 cpu",
        "bench_official_eth3d_vulkan.md:cpp:cpp f16 vulkan"])
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

    imgs, depths, Ks, poses, valids = [], [], [], [], []
    for idx in window:
        img, depth, K, pose = load_wai_view(scene_root, frames[idx])
        img, depth, K = crop_resize_if_necessary(
            image=img, resolution=(RW, RH), depthmap=depth,
            intrinsics=K, additional_quantities=None)
        imgs.append(np.asarray(img))
        depths.append(depth)
        Ks.append(K)
        poses.append(np.asarray(pose, dtype=np.float64))
        valids.append(depth > 1e-3)
    S = len(imgs)
    imgs_f = np.stack([np.transpose(im, (2, 0, 1)) for im in imgs]
                      ).astype(np.float32) / 255.0

    print("inference: torch ...")
    pose_t, depth_t = run_torch(args.pt, imgs_f, RH, RW)
    preds = {"torch": depth_t}
    poses_all = {"torch": pose_t}
    for q in QUANTS:
        gguf = Path(args.gguf_dir) / f"vggt-omega-1b-512-{q}.gguf"
        if not gguf.exists():
            print(f"skip {q}: {gguf.name} missing")
            continue
        print(f"inference: cpp {q} ...")
        pose_c, depth_c = run_cpp(args.cli, str(gguf), imgs_f, RH, RW)
        preds[q] = depth_c
        poses_all[q] = pose_c

    gt_pts = [unproject_depth(depths[i], Ks[i]) for i in range(S)]
    gt_factor = avg_dis_factor(gt_pts, valids)

    # scale-align each prediction set to GT (official avg_dis protocol)
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
            if pose_rows is None:
                T = np.linalg.inv(poses[0]) @ poses[i]
            else:
                T = pose_to_se3(pose_rows[i]).numpy()
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
        clouds[name] = [cloud_view0(preds[name], factors[name],
                                    poses_all[name])]
    chart_cloud(clouds, f"{scene}, {S} views",
                out_dir / "recon_pointcloud_comparison.png")

    # window metrics per source (identical functions as the eval matrix);
    # aligned depth is metric-scale, so divide by gt_factor to enter the
    # normalized space the GT side lives in (pr/pr_factor == aligned/gt_factor)
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

    # global official metrics from bench_official md files; entries accept
    # "file.md[:rename_torch[:rename_cpp]]" to give readable source labels
    official = {}
    for spec in args.official_md:
        parts = spec.split(":")
        p = Path(args.results_dir) / parts[0]
        if not p.exists():
            continue
        rename = {"torch": parts[1] if len(parts) > 1 else "torch",
                  "cpp": parts[2] if len(parts) > 2 else "cpp"}
        for m, srcs in parse_official_md(p).items():
            for src, v in srcs.items():
                official.setdefault(m, {})[rename.get(src, src)] = v
    # window fallback: only if the official z-depth metric is entirely
    # absent (e.g. official md files were not produced yet)
    if "z_depth_abs_rel" not in official:
        srcmap = {"torch": "torch", "f16": "cpp f16 cuda",
                  "q8_0": "cpp q8_0 cuda", "q6_K": "cpp q6_K cuda"}
        for name, r in rows.items():
            official.setdefault("z_depth_abs_rel", {})[
                srcmap.get(name, name)] = r["AbsRel"]

    chart_metrics(official, out_dir / "recon_metrics_comparison.png")

    keys = ["AbsRel", "SqRel", "RMSE", "RMSE-log", "d1", "accuracy",
            "completeness", "chamfer", "fscore"]
    lines = [f"# Reconstruction comparison — scene `{scene}`, window {window}"
             f", {RW}x{RH}", "",
             "| metric | " + " | ".join(rows.keys()) + " |",
             "|" + "--------|" * (len(rows) + 1)]
    for k in keys:
        lines.append(f"| {k} | " +
                     " | ".join(f"{rows[n].get(k, float('nan')):.6f}"
                                for n in rows) + " |")
    md = "\n".join(lines) + "\n"
    (out_dir / "recon_comparison.md").write_text(md)
    print(md)
    print("charts written:", *[p.name for p in sorted(
        out_dir.glob("recon_*"))], sep="\n  ")


if __name__ == "__main__":
    main()
