#!/usr/bin/env python3
"""Official MapAnything ETH3D dense-n-view benchmark, reproduced end-to-end
for both runtimes (official PyTorch wrapper vs the C++ ggml CLI).

This script walks the OFFICIAL path, not a reimplementation:
  - dataset:   mapanything.datasets.wai.eth3d.ETH3DWAI with the official test
               hyper-parameters (seed=777, num_views=2, covisibility_thres=0.025,
               resolution 512_1_52_ar = 512x336, transform='imgnorm') — i.e. the
               official random-walk covisibility sampling, 10 sets per scene;
  - model:     mapanything.models.external.vggt_omega.VGGTOmegaWrapper (torch),
               and for the C++ side the same MapAnything prediction contract
               with depth/pose_enc coming from vggt-cli (all derived quantities
               computed with the same helpers as the wrapper);
  - metrics:   get_all_info_for_metric_computation + the metric functions from
               benchmarking/dense_n_view/benchmark.py, producing the official
               8 metrics:
                 metric_scale_abs_rel, pointmaps_abs_rel,
                 pointmaps_inlier_thres_103, z_depth_abs_rel,
                 z_depth_inlier_thres_103, ray_dirs_err_deg, pose_ate_rmse,
                 pose_auc_5 (x100).

Official reference (vggt-omega-src/reproduction.md, MapAnything Dense-N-View
ETH3D, averaged over 2-100 input views, retrained checkpoint):
  AUC 79.533665 | ATE 0.009985055 | Point Abs 0.026264634 | Depth Abs 0.020428510

Usage (PYTHONPATH=/tmp/torch_cuda_lib:<repo>):
  python3 scripts/bench_official_eth3d.py \
      --data-root <...>/eth3d --metadata-dir <...>/metadata \
      [--cli build-cuda/bin/vggt-cli --gguf models/gguf/vggt-omega-1b-512-f16.gguf] \
      [--num-sets 130] [--out benchmarks/results/bench_official_eth3d.md]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CPP / "third_party" / "vggt-omega-src"))
sys.path.insert(0, str(REPO / "benchmarking" / "dense_n_view"))

from benchmark import get_all_info_for_metric_computation  # noqa: E402

# The official WAI io reads EXR via cv2, whose pip wheels silently fail on
# these files; imageio reads them fine. Depth EXRs are single-channel, so
# there is no BGB-order concern. This only patches the runtime in THIS
# process; the official repo code is untouched.
import mapanything.utils.wai.io as _wai_io  # noqa: E402

def _read_exr_imageio(fname, fmt="torch"):
    import imageio.v3 as _iio

    data = _iio.imread(str(fname)).astype(np.float32)
    if fmt == "torch":
        return torch.from_numpy(data)
    return data

_wai_io._read_exr = _read_exr_imageio

from mapanything.datasets.wai.eth3d import ETH3DWAI  # noqa: E402
from mapanything.models.external.vggt_omega import VGGTOmegaWrapper  # noqa: E402

OFFICIAL = {  # reproduction.md, retrained ckpt, 2-100 views average
    "pose_auc_5": 79.533665,
    "pose_ate_rmse": 0.009985055,
    "pointmaps_abs_rel": 0.026264634,
    "z_depth_abs_rel": 0.020428510,
}
METRICS = ["metric_scale_abs_rel", "pointmaps_abs_rel",
           "pointmaps_inlier_thres_103", "z_depth_abs_rel",
           "z_depth_inlier_thres_103", "ray_dirs_err_deg",
           "pose_ate_rmse", "pose_auc_5",
           # extension columns (not part of the official 8, reported for
           # diagnostics): mean relative rotation / camera-center error in the
           # view0 frame, and the rotation AUC@30 counterpart
           "rot_err_deg", "center_err", "rot_auc_30"]


def make_dataset(args):
    # "130 @ ETH3DWAI(...)" in the official config = ResizedDataset: resize to
    # 130 sets with per-index seed offsets (10 independent random walks per
    # scene); ResizedDataset requires set_epoch(0) before iteration.
    from mapanything.datasets.base.easy_dataset import ResizedDataset
    res = tuple(int(x) for x in args.resolution.split("x"))
    base = ETH3DWAI(
        resolution=res,                # official 512_1_52_ar = (512, 336)
        principal_point_centered=False,
        seed=777,                        # official test seed
        transform="imgnorm",
        data_norm_type="identity",
        ROOT=str(args.data_root),
        dataset_metadata_dir=str(args.metadata_dir),
        variable_num_views=False,
        num_views=args.num_views,
        covisibility_thres=0.025,
    )
    return ResizedDataset(args.num_sets, base)


def format_contract(pose_enc, depth, conf, H, W):
    """MapAnything prediction contract shared by both runtimes: pose_enc
    (B,S,9), depth/conf (B,S,H,W) on cuda -> list of per-view dicts."""
    from mapanything.utils.geometry import (
        convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
        convert_z_depth_to_depth_along_ray,
        depthmap_to_camera_frame,
        get_rays_in_camera_frame,
    )
    from vggt_omega.utils.geometry import closed_form_inverse_se3
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from vggt_omega.utils.rotation import mat_to_quat

    # identical formatting to VGGTOmegaWrapper._format_predictions (the
    # pose-encoding convention is shared: absT_quaR_FoV, FoV in radians,
    # principal point at the image center)
    extrinsics, intrinsics = encoding_to_camera(pose_enc, (H, W))
    out = []
    for v in range(pose_enc.shape[1]):
        world_from_camera = closed_form_inverse_se3(extrinsics[:, v])
        K = intrinsics[:, v]
        depth_z = depth[:, v]
        pts3d_cam, _ = depthmap_to_camera_frame(depth_z, K)
        depth_along_ray = convert_z_depth_to_depth_along_ray(
            depth_z, K).unsqueeze(-1)
        _, ray_dirs = get_rays_in_camera_frame(
            K, H, W, normalize_to_unit_sphere=True)
        cam_trans = world_from_camera[..., :3, 3]
        cam_quats = mat_to_quat(world_from_camera[..., :3, :3])
        pts3d = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
            ray_dirs, depth_along_ray, cam_trans, cam_quats)
        out.append({"pts3d": pts3d, "pts3d_cam": pts3d_cam,
                    "ray_directions": ray_dirs,
                    "depth_along_ray": depth_along_ray,
                    "cam_trans": cam_trans, "cam_quats": cam_quats,
                    "conf": conf[:, v]})
    return out


class VGGTOriginalWrapper:
    """Official facebook/VGGT-1B torch side for --arch vggt, exposing the
    same MapAnything contract as VGGTOmegaWrapper."""

    def __init__(self, src, ckpt):
        sys.path.insert(0, str(src))
        from vggt.models.vggt import VGGT
        model = VGGT().eval()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        model.load_state_dict(sd)
        self.model = model.cuda().eval()

    @torch.no_grad()
    def __call__(self, batch):
        imgs = torch.stack([v["img"] for v in batch], dim=1).cuda()
        B, S, _, H, W = imgs.shape
        preds = self.model(imgs)
        pose_enc = preds["pose_enc"]
        # official VGGT packs the channel LAST: depth (B,S,H,W,1)
        depth = preds["depth"]
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        conf = preds["depth_conf"]
        if conf.ndim == 5 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        return format_contract(pose_enc, depth.float(), conf.float(), H, W)


class MapAnythingLocalWrapper:
    """Official MapAnything torch side for --arch mapanything, loading the
    same local checkpoint dir the GGUF was converted from.  The dataset
    batches carry raw [0,1] images (transform='imgnorm'); the uniception
    encoder only ASSERTS the declared normalization and never converts, so
    the wrapper applies the ImageNet constants itself (matching the C++
    in-graph normalization) and declares 'dinov2'."""

    def __init__(self, ckpt_dir):
        from mapanything.models.mapanything.model import MapAnything
        self.model = MapAnything.from_pretrained(str(ckpt_dir)).cuda().eval()

    @torch.no_grad()
    def __call__(self, batch):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device="cuda").view(1, 3, 1, 1)
        imgs = torch.stack([v["img"] for v in batch], dim=1).cuda()
        imgs = (imgs - mean) / std
        S = imgs.shape[1]
        views = [{"img": imgs[:, v], "data_norm_type": ["dinov2"]}
                 for v in range(S)]
        res = self.model(views)
        keys = ("pts3d", "pts3d_cam", "ray_directions", "depth_along_ray",
                "cam_trans", "cam_quats", "conf", "metric_scaling_factor")
        return [{k: res[v][k] for k in keys} for v in range(S)]


class Pi3LocalWrapper:
    """Pi3 torch side for --arch pi3, following the MapAnything official
    Pi3Wrapper contract (mapanything/models/external/pi3/__init__.py) but
    loading from the local safetensors dir instead of the HF hub:
    pts3d_cam = local_points, depth_along_ray = ||local_points||,
    pts3d = the model's global points, conf = raw logits (no transform).
    Pi3 predicts no intrinsics, so the contract needs no FoV."""

    def __init__(self, ckpt_dir):
        sys.path.insert(0, str(CPP / "third_party" / "pi3-src"))
        from vggt_omega.utils.rotation import mat_to_quat
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained(str(ckpt_dir)).eval()
        self.model = model.cuda().eval()
        self._mat_to_quat = mat_to_quat

    @torch.no_grad()
    def __call__(self, batch):
        imgs = torch.stack([v["img"] for v in batch], dim=1).cuda()
        preds = self.model(imgs)
        ext = preds["camera_poses"]            # (B,N,4,4) camera-to-world
        cam_trans = ext[..., :3, 3]
        cam_quats = self._mat_to_quat(ext[..., :3, :3])
        pts3d_cam = preds["local_points"]      # (B,N,H,W,3)
        dal = torch.norm(pts3d_cam, dim=-1, keepdim=True)
        rays = pts3d_cam / dal
        n_views = ext.shape[1]
        return [{"pts3d": preds["points"][:, v],
                 "pts3d_cam": pts3d_cam[:, v],
                 "ray_directions": rays[:, v],
                 "depth_along_ray": dal[:, v],
                 "cam_trans": cam_trans[:, v],
                 "cam_quats": cam_quats[:, v],
                 "conf": preds["conf"][:, v]}
                for v in range(n_views)]


class Pi3XLocalWrapper:
    """Pi3X torch side for --arch pi3x.  Same contract as Pi3 above, but the
    official forward ALREADY multiplies local_points/points/cam_trans by the
    metric scale (pi3x.py decode_tail) and returns the exp'd factor as
    `metric` -- exposed here as metric_scaling_factor (optional key in the
    official benchmark)."""

    def __init__(self, ckpt_dir):
        sys.path.insert(0, str(CPP / "third_party" / "pi3-src"))
        from vggt_omega.utils.rotation import mat_to_quat
        from pi3.models.pi3x import Pi3X
        model = Pi3X.from_pretrained(str(ckpt_dir)).eval()
        self.model = model.cuda().eval()
        self._mat_to_quat = mat_to_quat

    @torch.no_grad()
    def __call__(self, batch):
        imgs = torch.stack([v["img"] for v in batch], dim=1).cuda()
        preds = self.model(imgs)
        ext = preds["camera_poses"]
        cam_trans = ext[..., :3, 3]
        cam_quats = self._mat_to_quat(ext[..., :3, :3])
        pts3d_cam = preds["local_points"]
        dal = torch.norm(pts3d_cam, dim=-1, keepdim=True)
        rays = pts3d_cam / dal
        msf = preds["metric"].float().reshape(1).cuda()
        n_views = ext.shape[1]
        return [{"pts3d": preds["points"][:, v],
                 "pts3d_cam": pts3d_cam[:, v],
                 "ray_directions": rays[:, v],
                 "depth_along_ray": dal[:, v],
                 "cam_trans": cam_trans[:, v],
                 "cam_quats": cam_quats[:, v],
                 "conf": preds["conf"][:, v],
                 "metric_scaling_factor": msf}
                for v in range(n_views)]


def procrustes_rigid(src, dst):
    """Closed-form rigid alignment (Horn): R,t minimizing
    sum_i || R x_i + t - y_i ||^2 for the known pixel-grid correspondences
    src=x (view1's own frame), dst=y (the same points in view0's frame).
    float64 SVD; returns (R, t, scale) with scale kept as a DIAGNOSTIC only
    (dust3r's two heads share one scale; pose uses R|t).  Shared by BOTH
    runtimes so the pose recovery is bit-identical on the two sides."""
    x = src.to(torch.float64)
    y = dst.to(torch.float64)
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    xc = x - mx
    yc = y - my
    # cross-covariance H = sum xc_i yc_i^T (3x3); SVD-based orthogonal
    # Procrustes with a det correction so R stays a proper rotation
    Hm = xc.T @ yc
    U, Sv, Vh = torch.linalg.svd(Hm)
    R0 = Vh.T @ U.T
    det = torch.det(R0)
    D = torch.diag(torch.tensor([1.0, 1.0, torch.sign(det).item()],
                                dtype=torch.float64, device=x.device))
    R = Vh.T @ D @ U.T
    t = my - R @ mx
    scale = (Sv * torch.diag(D)).sum() / (xc * xc).sum().clamp_min(1e-30)
    return R, t, scale


def assemble_dust3r(cam0_pts, common1_pts, cam1_pts, conf0, conf1):
    """Build the MapAnything prediction contract for --arch dust3r from the
    pair-wise network outputs (torch tensors, batch dim (B,...) or (...,)):
      cam0_pts   = head1 of run (view0, view1): view0 pts3d_cam (view0 frame)
      common1_pts= head2 of run (view0, view1): view1 pts in view0's frame
      cam1_pts   = head1 of run (view1, view0): view1 pts3d_cam (view1 frame)
    view0 pose = identity; view1 pose = rigid Procrustes aligning cam1_pts
    -> common1_pts (c2w of view1 in view0's frame).  Deterministic on both
    runtimes (same function, same inputs -> same pose)."""
    R, t, scale = procrustes_rigid(cam1_pts.reshape(-1, 3),
                                   common1_pts.reshape(-1, 3))
    Rf = R.to(torch.float32)
    tf = t.to(torch.float32)
    ext1 = torch.eye(4, dtype=torch.float32, device=Rf.device)
    ext1[:3, :3] = Rf
    ext1[:3, 3] = tf
    ext = torch.stack([torch.eye(4, dtype=torch.float32, device=Rf.device),
                       ext1]).unsqueeze(0)            # (1,2,4,4) c2w
    cam_trans = ext[..., :3, 3]
    from vggt_omega.utils.rotation import mat_to_quat
    cam_quats = mat_to_quat(ext[..., :3, :3])
    pts3d_cam = torch.stack([cam0_pts, cam1_pts])       # (2,H,W,3) or (B,2,...)
    pts3d = torch.stack([cam0_pts, common1_pts])        # both in view0 frame
    dal = torch.norm(pts3d_cam, dim=-1, keepdim=True)
    rays = pts3d_cam / dal.clamp_min(1e-8)
    conf = torch.stack([conf0, conf1])
    out = [{"pts3d": pts3d[v], "pts3d_cam": pts3d_cam[v],
            "ray_directions": rays[v], "depth_along_ray": dal[v],
            "cam_trans": cam_trans[:, v], "cam_quats": cam_quats[:, v],
            "conf": conf[v]} for v in range(2)]
    return out, float(scale.item())


class Dust3rLocalWrapper:
    """Official naver/dust3r torch side for --arch dust3r. The pair-wise
    model has NO pose head; poses are recovered in closed form (shared
    assemble_dust3r, see above) from two forwards:
      run A (view0, view1): head1_A = view0 pts3d_cam (view0 frame);
                            head2_A = view1 pts in view0's frame (common)
      run B (view1, view0): head1_B = view1 pts3d_cam (view1 frame)
    Inputs are [0,1] (dataset data_norm_type=identity); the official
    ImgNorm ((x-0.5)/0.5) is applied here, matching the C++ graph."""

    def __init__(self, ckpt_dir):
        sys.path.insert(0, str(CPP / "third_party" / "dust3r-src"))
        import dust3r.utils.path_to_croco  # noqa: F401
        from dust3r.model import AsymmetricCroCo3DStereo
        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            str(ckpt_dir)).cuda().eval()

    def _pair(self, im_a, im_b):
        shape = torch.tensor([[im_a.shape[-2], im_a.shape[-1]]],
                             device=im_a.device)
        return self.model(dict(img=im_a, true_shape=shape, instance="a"),
                          dict(img=im_b, true_shape=shape, instance="b"))

    @torch.no_grad()
    def __call__(self, batch):
        imgs = torch.stack([v["img"] for v in batch], dim=1).cuda() * 2.0 - 1.0
        assert imgs.shape[1] == 2, "dust3r is pair-wise"
        v0, v1 = imgs[:, 0], imgs[:, 1]
        resA1, resA2 = self._pair(v0, v1)
        resB1, _ = self._pair(v1, v0)
        # official forward renames head2's output key to pts3d_in_other_view
        preds, scale = assemble_dust3r(resA1["pts3d"],
                                       resA2["pts3d_in_other_view"],
                                       resB1["pts3d"],
                                       resA1["conf"], resB1["conf"])
        self.last_scale = scale
        return preds


@torch.no_grad()
def cpp_predictions(views, args):
    """Run vggt-cli on the batch images and format predictions through the
    same MapAnything prediction contract as the torch wrapper for --arch
    (vggt_omega/vggt: pose_enc (S,9) + z-depth through format_contract;
    pi3: 4x4 poses + local/global points + conf, the official Pi3Wrapper
    contract)."""
    imgs = torch.stack([v["img"] for v in views], dim=1)  # (B,S,3,H,W)
    B, S, _, H, W = imgs.shape
    with tempfile.TemporaryDirectory() as td:
        if args.arch == "dust3r":
            # pair-wise model: TWO CLI runs (A: v0,v1 and B: v1,v0 swap);
            # the shared assemble_dust3r recovers the pose identically on
            # both runtimes (no K, no GT, closed form)
            frames = imgs[0].cpu().contiguous().numpy()  # (2,3,H,W)
            raw = {}
            for tag, fr in (("a", frames), ("b", frames[::-1].copy())):
                fr.tofile(f"{td}/frames_{tag}.bin")
                prefix = f"{td}/out_{tag}"
                proc = subprocess.run(
                    [args.cli, "--model", args.gguf, "--bin",
                     f"{td}/frames_{tag}.bin", "--H", str(H), "--W", str(W),
                     "--S", "2", "--out-prefix", prefix],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace")
                if proc.returncode != 0:
                    raise RuntimeError(f"CLI failed:\n{proc.stderr[-800:]}")
                lp = np.fromfile(f"{prefix}.local_points.bin",
                                 dtype=np.float32).reshape(2, H, W, 3)
                cf = np.fromfile(f"{prefix}.conf.bin",
                                 dtype=np.float32).reshape(2, H, W, 1)
                raw[tag] = (lp, cf)
            t0 = lambda a: torch.from_numpy(a).cuda().unsqueeze(0)  # noqa: E731  (B,...)
            preds, scale = assemble_dust3r(
                t0(raw["a"][0][0]), t0(raw["a"][0][1]), t0(raw["b"][0][0]),
                t0(raw["a"][1][0]),      # run A head1 conf (B,H,W,1)
                t0(raw["b"][1][0]))      # run B head1 = cam1 self
            return preds
        frames = imgs[0].cpu().contiguous().numpy()  # (S,3,H,W)
        frames.tofile(f"{td}/frames.bin")
        prefix = f"{td}/out"
        proc = subprocess.run(
            [args.cli, "--model", args.gguf, "--bin", f"{td}/frames.bin",
             "--H", str(H), "--W", str(W), "--S", str(S),
             "--out-prefix", prefix], capture_output=True, text=True,
                    encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"CLI failed:\n{proc.stderr[-800:]}")
        if args.arch in ("pi3", "pi3x"):
            ext = torch.from_numpy(
                np.fromfile(f"{prefix}.pose.bin", dtype=np.float32)
                .reshape(S, 4, 4)).unsqueeze(0).cuda()
            lp = torch.from_numpy(
                np.fromfile(f"{prefix}.local_points.bin", dtype=np.float32)
                .reshape(S, H, W, 3)).unsqueeze(0).cuda()
            conf = torch.from_numpy(
                np.fromfile(f"{prefix}.conf.bin", dtype=np.float32)
                .reshape(S, H, W, 1)).unsqueeze(0).cuda()
            pts = torch.from_numpy(
                np.fromfile(f"{prefix}.points.bin", dtype=np.float32)
                .reshape(S, H, W, 3)).unsqueeze(0).cuda()
            scale = (float(np.fromfile(f"{prefix}.scale.bin",
                                       dtype=np.float32)[0])
                     if args.arch == "pi3x" else None)
        elif args.arch == "mapanything":
            ext = torch.from_numpy(
                np.fromfile(f"{prefix}.pose.bin", dtype=np.float32)
                .reshape(S, 4, 4)).unsqueeze(0).cuda()
            rays = torch.from_numpy(
                np.fromfile(f"{prefix}.local_points.bin", dtype=np.float32)
                .reshape(S, H, W, 3)).unsqueeze(0).cuda()  # unit-sphere dirs
            dal = torch.from_numpy(
                np.fromfile(f"{prefix}.depth.bin", dtype=np.float32)
                .reshape(S, H, W)).unsqueeze(0).cuda()  # already x scale
            conf = torch.from_numpy(
                np.fromfile(f"{prefix}.conf.bin", dtype=np.float32)
                .reshape(S, H, W, 1)).unsqueeze(0).cuda()
            pts = torch.from_numpy(
                np.fromfile(f"{prefix}.points.bin", dtype=np.float32)
                .reshape(S, H, W, 3)).unsqueeze(0).cuda()
            scale = float(np.fromfile(f"{prefix}.scale.bin",
                                      dtype=np.float32)[0])
        else:
            pose_enc = torch.from_numpy(
                np.fromfile(f"{prefix}.pose.bin", dtype=np.float32)
                .reshape(S, 9)).unsqueeze(0).cuda()  # (1,S,9)
            depth = torch.from_numpy(
                np.fromfile(f"{prefix}.depth.bin", dtype=np.float32)
                .reshape(S, H, W)).unsqueeze(0).cuda()  # (1,S,H,W)
            conf = torch.from_numpy(
                np.fromfile(f"{prefix}.depth_conf.bin", dtype=np.float32)
                .reshape(S, H, W)).unsqueeze(0).cuda()

    if args.arch in ("pi3", "pi3x"):
        from vggt_omega.utils.rotation import mat_to_quat
        cam_trans = ext[..., :3, 3]
        cam_quats = mat_to_quat(ext[..., :3, :3])
        dal = torch.norm(lp, dim=-1, keepdim=True)
        rays = lp / dal
        msf = (torch.full((1,), scale, dtype=torch.float32, device="cuda")
               if scale is not None else None)
        return [{"pts3d": pts[:, v], "pts3d_cam": lp[:, v],
                 "ray_directions": rays[:, v],
                 "depth_along_ray": dal[:, v],
                 "cam_trans": cam_trans[:, v],
                 "cam_quats": cam_quats[:, v],
                 "conf": conf[:, v],
                 **({"metric_scaling_factor": msf} if msf is not None else {})}
                for v in range(S)]
    if args.arch == "mapanything":
        # The official MapAnything contract: local_points are UNIT ray dirs,
        # pts3d_cam = ray_dirs * depth_along_ray; the C++ .depth/.points are
        # already multiplied by the metric scale, matching the official
        # adaptor outputs (cam_trans likewise; rays and quats unscaled).
        from vggt_omega.utils.rotation import mat_to_quat
        cam_trans = ext[..., :3, 3]
        cam_quats = mat_to_quat(ext[..., :3, :3])
        pts3d_cam = rays * dal[..., None]
        msf = torch.full((1,), scale, dtype=torch.float32,
                         device="cuda")  # (B,) shared across views
        return [{"pts3d": pts[:, v], "pts3d_cam": pts3d_cam[:, v],
                 "ray_directions": rays[:, v],
                 "depth_along_ray": dal[:, v, None],
                 "cam_trans": cam_trans[:, v],
                 "cam_quats": cam_quats[:, v],
                 "conf": conf[:, v],
                 "metric_scaling_factor": msf} for v in range(S)]
    return format_contract(pose_enc, depth, conf, H, W)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--cli", default=str(CPP / "build-cuda/bin/vggt-cli"))
    ap.add_argument("--gguf", default=str(CPP / "models/gguf/"
                                          "vggt-omega-1b-512-f16.gguf"))
    ap.add_argument("--ckpt", default=str(CPP / "models/pytorch/"
                                          "vggt_omega_1b_512.pt"))
    ap.add_argument("--arch", default="vggt_omega",
                    choices=["vggt_omega", "vggt", "pi3", "pi3x",
                             "mapanything", "dust3r"],
                    help="torch-side model; vggt = official facebook/VGGT-1B "
                         "and pi3 = official yyfz233/Pi3 (both patch 14: use "
                         "a /14 resolution, e.g. 518x392); dust3r = official "
                         "naver/dust3r (patch 16 pair-wise: 512x336 works, "
                         "poses recovered by closed-form Procrustes)")
    ap.add_argument("--vggt-src", default="/tmp/vggt-src")
    ap.add_argument("--vggt-ckpt", default=str(CPP / "models/pytorch/"
                                               "vggt1b/model.pt"))
    ap.add_argument("--pi3-ckpt", default=str(CPP / "models/pytorch/pi3"))
    ap.add_argument("--pi3x-ckpt", default=str(CPP / "models/pytorch/pi3x"))
    ap.add_argument("--ma-ckpt", default=str(CPP / "models/pytorch/mapanything"),
                    help="--arch mapanything: local checkpoint dir (same "
                         "source the GGUF was converted from)")
    ap.add_argument("--dust3r-ckpt", default=str(CPP / "models/pytorch/dust3r"),
                    help="--arch dust3r: local HF layout dir (config.json + "
                         "model.safetensors)")
    ap.add_argument("--resolution", default="512x336",
                    help="dataset resize tuple (W, H), same order as the "
                         "official (512, 336)")
    ap.add_argument("--num-views", type=int, default=2)
    ap.add_argument("--num-sets", type=int, default=130,
                    help="official config resizes the test set to 130")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--skip-cpp", action="store_true")
    ap.add_argument("--out", default=str(CPP / "benchmarks/results/"
                                         "bench_official_eth3d.md"))
    args = ap.parse_args()

    torch.manual_seed(777)
    np.random.seed(777)

    dataset = make_dataset(args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2)
    loader.dataset.set_epoch(0)
    n_sets = len(loader)
    print(f"official ETH3D protocol: {n_sets} sets x {args.num_views} views, "
          f"{args.resolution}, seed 777 (official random-walk resize)")

    torch_model = None
    if not args.skip_torch:
        if args.arch == "vggt":
            torch_model = VGGTOriginalWrapper(args.vggt_src, args.vggt_ckpt)
        elif args.arch == "pi3":
            torch_model = Pi3LocalWrapper(args.pi3_ckpt)
        elif args.arch == "pi3x":
            torch_model = Pi3XLocalWrapper(args.pi3x_ckpt)
        elif args.arch == "mapanything":
            torch_model = MapAnythingLocalWrapper(args.ma_ckpt)
        elif args.arch == "dust3r":
            torch_model = Dust3rLocalWrapper(args.dust3r_ckpt)
        else:
            torch_model = VGGTOmegaWrapper(
                "vggt_omega", False, args.ckpt).cuda().eval()

    agg = {src: {m: [] for m in METRICS} for src in
           ([] if args.skip_torch else ["torch"]) +
           ([] if args.skip_cpp else ["cpp"])}
    for it, batch in enumerate(loader):
        n_views = len(batch)
        for view in batch:
            view["idx"] = view["idx"][2:]
        ignore = {"depthmap", "dataset", "label", "instance", "idx",
                  "true_shape", "rng", "data_norm_type"}
        for view in batch:
            for k in view.keys():
                if k not in ignore:
                    view[k] = view[k].cuda(non_blocking=True)

        preds_by_src = {}
        if torch_model is not None:
            with torch.no_grad():
                preds_by_src["torch"] = torch_model(batch)
        if not args.skip_cpp:
            preds_by_src["cpp"] = cpp_predictions(batch, args)

        for src, preds in preds_by_src.items():
            gt_info, pr_info, valid_masks = (
                get_all_info_for_metric_computation(batch, preds))
            batch_size = batch[0]["img"].shape[0]
            for b in range(batch_size):
                vals = {m: [] for m in
                        ("pointmaps_abs_rel", "pointmaps_inlier_thres_103",
                         "z_depth_abs_rel", "z_depth_inlier_thres_103",
                         "ray_dirs_err_deg")}
                gt_poses, pr_poses = [], []
                for v in range(n_views):
                    vm = valid_masks[v][b].numpy()
                    vals["pointmaps_abs_rel"].append(__import__(
                        "mapanything.utils.metrics", fromlist=["m_rel_ae"]
                    ).m_rel_ae(gt_info["pts3d"][v][b].numpy(),
                               pr_info["pts3d"][v][b].numpy(), vm))
                    from mapanything.utils.metrics import (  # noqa: E402
                        calculate_auc_np, evaluate_ate, thresh_inliers,
                        se3_to_relative_pose_error,
                        l2_distance_of_unit_ray_directions_to_angular_error)
                    vals["pointmaps_inlier_thres_103"].append(thresh_inliers(
                        gt=gt_info["pts3d"][v][b].numpy(),
                        pred=pr_info["pts3d"][v][b].numpy(),
                        mask=vm, thresh=1.03))
                    vals["z_depth_abs_rel"].append(__import__(
                        "mapanything.utils.metrics", fromlist=["m_rel_ae"]
                    ).m_rel_ae(gt_info["z_depths"][v][b].numpy(),
                               pr_info["z_depths"][v][b].numpy(), vm))
                    vals["z_depth_inlier_thres_103"].append(thresh_inliers(
                        gt=gt_info["z_depths"][v][b].numpy(),
                        pred=pr_info["z_depths"][v][b].numpy(),
                        mask=vm, thresh=1.03))
                    l2 = torch.norm(
                        gt_info["ray_directions"][v][b]
                        - pr_info["ray_directions"][v][b], dim=-1)
                    vals["ray_dirs_err_deg"].append(float(
                        l2_distance_of_unit_ray_directions_to_angular_error(
                            l2).mean()))
                    gt_poses.append(gt_info["poses"][v][b])
                    pr_poses.append(pr_info["poses"][v][b])
                ate = evaluate_ate(gt_traj=gt_poses, est_traj=pr_poses)
                gt_s = torch.stack(gt_poses)
                pr_s = torch.stack(pr_poses)
                rr, tt = se3_to_relative_pose_error(pr_s, gt_s, n_views)
                auc5 = calculate_auc_np(rr.cpu().numpy(), tt.cpu().numpy(),
                                        max_threshold=5)[0] * 100.0
                auc30 = calculate_auc_np(rr.cpu().numpy(), tt.cpu().numpy(),
                                         max_threshold=30)[0] * 100.0
                agg[src]["pointmaps_abs_rel"].append(
                    float(np.mean(vals["pointmaps_abs_rel"])))
                agg[src]["pointmaps_inlier_thres_103"].append(
                    float(np.mean(vals["pointmaps_inlier_thres_103"])))
                agg[src]["z_depth_abs_rel"].append(
                    float(np.mean(vals["z_depth_abs_rel"])))
                agg[src]["z_depth_inlier_thres_103"].append(
                    float(np.mean(vals["z_depth_inlier_thres_103"])))
                agg[src]["ray_dirs_err_deg"].append(
                    float(np.mean(vals["ray_dirs_err_deg"])))
                agg[src]["pose_ate_rmse"].append(float(
                    np.asarray(ate).ravel()[0]))
                agg[src]["pose_auc_5"].append(float(auc5))
                agg[src]["rot_err_deg"].append(float(rr.mean().cpu()))
                agg[src]["center_err"].append(float(tt.mean().cpu()))
                agg[src]["rot_auc_30"].append(float(auc30))
                # metric scale (only when GT factor is valid)
                if gt_info["metric_scale"] is not None:
                    gtn = gt_info["metric_scale"][b].item()
                    prn = pr_info["metric_scale"][b].item()
                    agg[src]["metric_scale_abs_rel"].append(
                        abs(prn - gtn) / gtn)
        if (it + 1) % 20 == 0 or it + 1 == n_sets:
            print(f"  {it + 1}/{n_sets} sets done")

    lines = ["# Official MapAnything ETH3D protocol (random-walk sampling, "
             f"seed 777, {args.num_views} views, {args.resolution}, "
             f"arch={args.arch}, torch f32 vs cpp gguf)", "",
             f"sets evaluated: {n_sets} (10 per scene x 13 scenes)", "",
             "Official reference (reproduction.md, retrained ckpt, 2-100 "
             "views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 "
             "/ Depth Abs 0.020429", "",
             "| metric | " + " | ".join(agg.keys()) + " |",
             "|" + "--------|" * (len(agg) + 1)]
    for m in METRICS:
        row = [m]
        for src in agg:
            row.append(f"{np.mean(agg[src][m]):.6f}" if agg[src][m] else "—")
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines) + "\n"
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
