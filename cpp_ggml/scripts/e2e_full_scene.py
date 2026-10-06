#!/usr/bin/env python3
"""Full-scene end-to-end reconstruction: ALL frames of a WAI scene through
the official torch demo path AND the ggml CLI, chunked (sliding window),
aligned and merged into ONE global colored point cloud per runtime.

Why chunked: the vggt-omega family runs global attention over all view
tokens, so a whole 38-frame scene cannot fit in one forward on a busy
CPU/GPU box.

Two alignment modes (--align):
  chain (legacy) : Horn Sim3 on the 2 shared frames per boundary, chained
                   chunk-to-chunk. DEPRECATED for quality: each chunk gets
                   its own global scale basis from the model, and the
                   shared frames' depth differs non-rigidly between the
                   two contexts, so the per-boundary Sim3 (scales up to
                   1.56 were observed) leaves residual misalignment that
                   ACCUMULATES along the chain (walls visibly split in
                   the fused cloud).
  gt (default)   : benchmark scenes ship metric GT camera poses; each
                   chunk is anchored INDEPENDENTLY into the GT world
                   frame with one robust Horn Sim3 fitted on the 8 model
                   camera centers vs the 8 GT camera centers. No chain,
                   no accumulation; residual = per-chunk pose error only.

BOTH runtimes use the identical chunking/alignment code, so the fused-cloud
comparison is unaffected by drift (both drift the same).

Usage:
  PYTHONPATH=/tmp/torch_cuda_lib:. python3 scripts/e2e_full_scene.py \
      --scene-dir ../data/map-anything-benchmarking/eth3d/courtyard \
      --out-dir benchmarks/results/vggt-omega/e2e_courtyard_full \
      --gguf models/gguf/vggt-omega-1b-512-f16.gguf \
      --cli build-cpu/bin/vggt-cli --device cpu
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "scripts"))
from export_recon_ply import write_ply  # noqa: E402


def horn_sim3(src, dst):
    """Closed-form Sim3 (s, R, t) with dst ~= s * R @ src + t (Horn)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    var = (sc ** 2).sum() / len(src)
    s = np.trace(D * S) / var if var > 0 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def robust_sim3(src, dst, iters=2):
    """Horn Sim3 with outlier rejection (drop > median + 3*MAD residuals)."""
    keep = np.ones(len(src), bool)
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(iters):
        s, R, t = horn_sim3(src[keep], dst[keep])
        res = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
        med = np.median(res[keep])
        mad = np.median(np.abs(res[keep] - med)) + 1e-12
        keep = res <= med + 3 * 1.4826 * mad
    return s, R, t, res, keep


def apply_sim3(pts, s, R, t):
    return s * (pts @ R.T) + t


def voxel_down(pts, cols, confs, vsize):
    """Confidence-priority voxel merge: per voxel keep the HIGHEST-conf
    observation (the official viz concatenates instead, which leaves the
    multi-view depth-noise band visible as blur; picking the best view per
    voxel sharpens surfaces while staying deterministic)."""
    key = np.floor(pts / vsize).astype(np.int64)
    order = np.argsort(-confs)  # descending: first hit per voxel = max conf
    _, first = np.unique(key[order], axis=0, return_index=True)
    idx = np.sort(order[first])
    return pts[idx], cols[idx]


def depth_edge(depth, rtol=0.03, kernel_size=3):
    """Official visual_util.depth_edge: pixels whose 3x3 depth window jumps
    more than rtol are depth discontinuities (flying pixels at object
    boundaries) and must not enter the cloud."""
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])
    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y: y + depth.shape[-2], x: x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    rel = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (rel > rtol).reshape(original_shape)


def official_world_points(depth, pose_enc, hw):
    """The demo's own world_points_from_depth formula (w2c convention)."""
    h, w = hw
    ext = np.zeros((len(pose_enc), 4, 4))
    K = np.zeros((len(pose_enc), 3, 3))
    for i, p in enumerate(pose_enc):
        quat = p[3:7] / np.linalg.norm(p[3:7])
        R = np.array([
            [1 - 2 * (quat[1] ** 2 + quat[2] ** 2),
             2 * (quat[0] * quat[1] - quat[2] * quat[3]),
             2 * (quat[0] * quat[2] + quat[1] * quat[3])],
            [2 * (quat[0] * quat[1] + quat[2] * quat[3]),
             1 - 2 * (quat[0] ** 2 + quat[2] ** 2),
             2 * (quat[1] * quat[2] - quat[0] * quat[3])],
            [2 * (quat[0] * quat[2] - quat[1] * quat[3]),
             2 * (quat[1] * quat[2] + quat[0] * quat[3]),
             1 - 2 * (quat[0] ** 2 + quat[1] ** 2)]])
        ext[i, :3, :3], ext[i, :3, 3], ext[i, 3, 3] = R, p[0:3], 1.0
        K[i, 0, 0] = (w / 2) / np.tan(p[8] / 2)
        K[i, 1, 1] = (h / 2) / np.tan(p[7] / 2)
        K[i, 0, 2], K[i, 1, 2], K[i, 2, 2] = w / 2, h / 2, 1.0
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = np.broadcast_to(x[None], (len(pose_enc), h, w))
    y = np.broadcast_to(y[None], (len(pose_enc), h, w))
    fx, fy = K[:, 0, 0][:, None, None], K[:, 1, 1][:, None, None]
    cx, cy = K[:, 0, 2][:, None, None], K[:, 1, 2][:, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], -1)
    rot, tr = ext[:, :3, :3], ext[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rot, (0, 2, 1)),
                     cam - tr[:, None, None, :]), ext[:, :3, :], K


def torch_chunk(images_paths, args):
    """Official demo forward for one chunk (returns per-frame world/depth/conf/rgb)."""
    import torch
    sys.path.insert(0, str(CPP / "third_party" / "vggt-omega-src"))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    model = VGGTOmega().eval()
    sd = torch.load(CPP / args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    keys = set(model.state_dict().keys())
    model.load_state_dict({k: v for k, v in sd.items() if k in keys})
    model = model.to(args.device)
    imgs = load_and_preprocess_images(images_paths, mode="balanced",
                                      image_resolution=args.image_size)
    imgs = imgs.to(args.device)
    with torch.inference_mode():
        p = model(imgs)
    out = {}
    for k, v in p.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().cpu().numpy()
            if v.shape[0] == 1:
                v = v[0]  # drop the batch dim (official demo convention)
            out[k] = v
    model = model.cpu()
    return out


def cpp_chunk(images_paths, args, out_prefix):
    cmd = [str(CPP / args.cli), "--model", str(CPP / args.gguf)]
    for im in images_paths:
        cmd += ["--images", im]
    cmd += ["--image-size", str(args.image_size), "--out-prefix", out_prefix]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CPP))
    if r.returncode != 0:
        raise RuntimeError(f"CLI failed: {r.stderr[-1500:]}")
    s = len(images_paths)
    h = w = None
    meta = np.array(1)
    return out_prefix, s


def read_cpp(prefix, s, h, w):
    pose = np.fromfile(f"{prefix}.pose.bin", dtype=np.float32).reshape(s, 9)
    depth = np.fromfile(f"{prefix}.depth.bin",
                        dtype=np.float32).reshape(s, h, w)
    conf = np.fromfile(f"{prefix}.depth_conf.bin",
                       dtype=np.float32).reshape(s, h, w)
    return pose, depth, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gguf", default="models/gguf/vggt-omega-1b-512-f16.gguf")
    ap.add_argument("--cli", default="build-cuda/bin/vggt-cli")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ckpt", default="models/pytorch/vggt-omega/vggt_omega_1b_512.pt")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=0,
                    help="frames per forward; 0 = SINGLE PASS over all "
                         "frames — the official demo semantics and the "
                         "DEFAULT: 38 frames peak at only 8.75 GiB on a "
                         "24 GB card (~118 MiB/frame, so 100+ frames fit). "
                         "Set K>0 only on CPU/small-VRAM boxes")
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--conf-thres", type=float, default=3.0,
                    help="official gradio default depth_conf floor")
    ap.add_argument("--conf-percentile", type=float, default=20.0,
                    help="official gradio conf_thres=20: drop the lowest P%% "
                         "of confidences scene-wide (visual_util also "
                         "clamps the floor at absolute conf 2.0)")
    ap.add_argument("--depth-edge-rtol", type=float, default=0.03,
                    help="official depth_edge rtol (0 disables the edge "
                         "filter)")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="merge voxel size in model units (0 = off)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="debug cap on the number of frames (0 = all)")
    ap.add_argument("--tag", default="f16")
    ap.add_argument("--align", default="gt", choices=["chain", "gt"],
                    help="chain = legacy per-boundary Sim3 (accumulates); "
                         "gt = anchor every chunk independently into the "
                         "GT world frame (WAI scene_meta transform_matrix)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(args.scene_dir, "images").glob("*"))
    if args.max_frames:
        frames = frames[:args.max_frames]
    n = len(frames)
    if args.chunk == 0 or args.chunk >= n:
        args.chunk = n
        starts = [0]
    else:
        step = args.chunk - args.overlap
        starts = list(range(0, n, step))
        if starts[-1] != n - args.chunk and n - args.chunk > starts[-1]:
            starts.append(n - args.chunk)
        starts = sorted(set(min(s, max(0, n - args.chunk)) for s in starts))
    print(f"frames: {n}, chunks (K={args.chunk}, "
          f"{'single pass' if len(starts) == 1 else f'overlap={args.overlap}'}): "
          f"{len(starts)} -> {starts}")

    # ---- per-chunk inference on both runtimes
    chunks = []  # list of dict(frames_idx, world_t, conf_t, rgb_t, world_c, conf_c)
    h = w = None
    for ci, st in enumerate(starts):
        idx = list(range(st, min(st + args.chunk, n)))
        paths = [str(frames[i]) for i in idx]
        print(f"== chunk {ci}: frames {idx}")

        t = torch_chunk(paths, args)
        depth_t = t["depth"][..., 0]
        pose_t = t["pose_enc"].reshape(len(idx), 9)
        h, w = depth_t.shape[1:3]
        rgb = t["images"]
        if rgb.ndim == 4 and rgb.shape[1] == 3:
            rgb = rgb.transpose(0, 2, 3, 1)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        world_t, ext_t, _K = official_world_points(depth_t, pose_t, (h, w))

        prefix, _s = cpp_chunk(paths, args, str(out / "chunks" / f"cpp_{ci}"))
        pose_c, depth_c, conf_c = read_cpp(prefix, len(idx), h, w)
        world_c, ext_c, _Kc = official_world_points(depth_c, pose_c, (h, w))

        chunks.append(dict(idx=idx, names=[Path(p).stem for p in paths],
                           world_t=world_t, conf_t=t["depth_conf"],
                           depth_t=t["depth"][..., 0],
                           rgb_t=rgb, world_c=world_c, conf_c=conf_c,
                           ext_t=ext_t, ext_c=ext_c))

    # ---- official per-frame quality filters (visual_util.predictions_to_glb):
    # isfinite + depth-edge removal + (percentile | absolute) conf floor,
    # computed ONCE from the torch reference conf so both runtimes keep the
    # exact same pixel set (their own confs differ numerically).
    if args.conf_percentile > 0:
        allc = np.concatenate([ch["conf_t"].reshape(-1) for ch in chunks])
        # official visual_util.predictions_to_glb: floor = max(2.0, p)
        conf_floor = max(2.0, float(np.percentile(allc[np.isfinite(allc)],
                                                  args.conf_percentile)))
        print(f"official percentile filter: conf floor = {conf_floor:.3f} "
              f"(p{args.conf_percentile}, min 2.0)")
    else:
        conf_floor = args.conf_thres
    for ch in chunks:
        m = np.isfinite(ch["conf_t"]) & np.isfinite(ch["depth_t"])
        if args.depth_edge_rtol > 0:
            m &= ~depth_edge(ch["depth_t"], args.depth_edge_rtol)
        m &= ch["conf_t"] >= conf_floor
        ch["frame_masks"] = {f: m[i].reshape(-1)
                             for i, f in enumerate(ch["idx"])}

    # ---- per-chunk global alignment (both sides share the code path)
    align_stats = []  # (mode, ci, label, med, p95, scale)
    if args.align == "gt":
        meta = json.loads((Path(args.scene_dir) / "scene_meta.json")
                          .read_text())
        gt = {f["frame_name"]: np.array(f["transform_matrix"])[:3, 3]
              for f in meta["frames"]}
        for ci, ch in enumerate(chunks):
            for side in ("torch", "cpp"):
                ext = ch["ext_t" if side == "torch" else "ext_c"]
                # model camera centers in the chunk's own world frame
                src = np.stack([-ext[i][:3, :3].T @ ext[i][:3, 3]
                                for i in range(len(ch["idx"]))])
                dst = np.stack([gt[ch["names"][i]]
                                for i in range(len(ch["idx"]))])
                s, R, tt, res, _k = robust_sim3(src, dst)
                ch[f"sim3_{side}"] = (s, R, tt)
                align_stats.append(("gt-anchor", ci,
                                    f"{side} frames {ch['idx'][0]}-"
                                    f"{ch['idx'][-1]}",
                                    float(np.median(res)),
                                    float(np.percentile(res, 95)), s))

    def fuse(side):
        s_key = "world_t" if side == "torch" else "world_c"
        all_pts, all_cols, all_confs = [], [], []
        covered = set()
        for ci, ch in enumerate(chunks):
            pts = ch[s_key].astype(np.float64)
            if args.align == "gt":
                s, R, tt = ch[f"sim3_{side}"]
            elif ci == 0:
                s, R, tt = 1.0, np.eye(3), np.zeros(3)
            else:
                shared = np.intersect1d(ch["idx"], chunks[ci - 1]["idx"])
                a_idx = [ch["idx"].index(f) for f in shared]
                b_idx = [chunks[ci - 1]["idx"].index(f) for f in shared]
                src = pts[a_idx].reshape(-1, 3)
                dst = chunks[ci - 1][s_key].astype(
                    np.float64)[b_idx].reshape(-1, 3)
                s, R, tt, res, _k = robust_sim3(src, dst)
                align_stats.append(("chain", ci, str([int(x) for x in shared]),
                                    float(np.median(res)),
                                    float(np.percentile(res, 95)), s))
            new_f = [f for f in ch["idx"] if f not in covered]
            n_pos = [ch["idx"].index(f) for f in new_f]
            fused = apply_sim3(pts[n_pos].reshape(-1, 3), s, R, tt)
            rgb = np.broadcast_to(ch["rgb_t"],
                                  ch[s_key].shape[:-1] + (3,))
            rgb = rgb[n_pos].reshape(-1, 3)
            cf = np.concatenate(
                [ch["conf_t"].reshape(len(ch["idx"]), -1)[
                    ch["idx"].index(f)] for f in new_f])
            keep = np.concatenate([ch["frame_masks"][f] for f in new_f])
            all_pts.append(fused[keep])
            all_cols.append(rgb[keep])
            all_confs.append(cf[keep])
            covered.update(ch["idx"])
        pts = np.concatenate(all_pts)
        cols = np.concatenate(all_cols)
        confs = np.concatenate(all_confs)
        vox = voxel_down(pts, cols, confs, args.voxel) if args.voxel > 0 \
            else (pts, cols)
        return pts.astype(np.float32), cols, vox[0].astype(np.float32), \
            vox[1]

    pt_t, cl_t, vox_t, voxc_t = fuse("torch")
    pt_c, cl_c, vox_c, voxc_c = fuse("cpp")
    write_ply(out / f"torch_f32_global_{n}f_{args.tag}.ply", vox_t, voxc_t)
    write_ply(out / f"cpp_{args.tag}_global_{n}f.ply", vox_c, voxc_c)

    # ---- cross-runtime comparison on the fused clouds (same pixel grid)
    lines = [f"# Full-scene e2e reconstruction — {n} frames of "
             f"`{Path(args.scene_dir).name}`", "",
             f"- chunking: "
             f"{'SINGLE PASS (official semantics)' if len(starts) == 1 else f'K={args.chunk}, overlap={args.overlap}'}"
             f", {len(starts)} forward(s) per runtime; both sides share the "
             f"identical chunking + alignment code (align={args.align})",
             f"- merge: official filters (isfinite + depth_edge rtol="
             f"{args.depth_edge_rtol} + "
             f"{'conf percentile p' + str(args.conf_percentile) if args.conf_percentile > 0 else f'conf>={args.conf_thres}'}"
             f"), conf-priority voxel={args.voxel}", "",
             "| pair | points | mean | median | p95 | max |", "|---|---|---|---|---|---|"]
    for name, a, b in ((f"cpp {args.tag} vs torch f32", pt_t, pt_c),):
        d = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=1)
        lines.append(f"| {name} | {len(d):,} | {d.mean():.5f} | "
                     f"{np.median(d):.5f} | {np.percentile(d, 95):.5f} | "
                     f"{d.max():.5f} |")
        print(f"{name}: mean {d.mean():.5f} median {np.median(d):.5f}")
    lines += ["", f"## {'GT-anchor' if args.align == 'gt' else 'Chunk-boundary'} "
              "Sim3 fits", "",
              "| chunk | side / shared frames | median res | p95 res | scale |",
              "|---|---|---|---|---|"]
    for mode, ci, label, med, p95, s in align_stats:
        lines.append(f"| {ci} | {label} | {med:.5f} | {p95:.5f} | "
                     f"{s:.5f} |")
    if args.align == "gt":
        lines += ["", "(gt mode: residual = model camera centers vs GT "
                  "camera centers after the per-chunk Sim3, in GT meters; "
                  "scale maps each chunk's own model basis to the metric "
                  "GT frame — chunk-to-chunk variation of `scale` is the "
                  "model's cross-chunk scale drift, absorbed here instead "
                  "of accumulating through a chain.)"]
    (out / "ALIGNMENT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'ALIGNMENT.md'}")


if __name__ == "__main__":
    main()
