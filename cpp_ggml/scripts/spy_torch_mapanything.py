#!/usr/bin/env python3
"""Dump MapAnything stage tensors for parity debugging against the C++
builder.

Usage (PYTHONPATH=/tmp/torch_cuda_lib:<repo>):
  python3 scripts/spy_torch_mapanything.py <frames.bin> <H> <W> <S> <out_dir>

All token-stage dumps are f32 token-major (T, C) per frame batch (B=1);
conv-grid dumps keep the torch BCHW layout; final outputs are (S, H, W, ch)
physical.  Note: the official DINOv2 interpolate_pos_encoding ASSERTS on
non-square/non-trained sizes when interpolate_offset != 0 — this script
monkey-patches that assert away (numerics unchanged).

IMPORTANT: the model does NOT normalize images internally (uniception takes
already-normalized input); this script applies the ImageNet constants.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # CPU-torch
# fallbacks (plain python3 envs) must still produce a valid reference


REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from mapanything.models.mapanything.model import MapAnything  # noqa: E402

frames_bin, H, W, S, out_dir = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                int(sys.argv[4]), sys.argv[5])
outdir = Path(out_dir)
outdir.mkdir(parents=True, exist_ok=True)

manifest = []


def save(name, arr):
    arr = np.ascontiguousarray(arr.detach().float().cpu().numpy())
    arr.tofile(outdir / f"{name}.bin")
    md5 = hashlib.md5((outdir / f"{name}.bin").read_bytes()).hexdigest()[:10]
    manifest.append(f"{name} {arr.shape} {md5}")
    print(f"  {name} {arr.shape} {md5}", flush=True)


frames = np.fromfile(frames_bin, dtype=np.float32).reshape(S, 3, H, W)
model = MapAnything.from_pretrained(
    str(REPO / "cpp_ggml/models/pytorch/mapanything")).eval().to(DEV).float()

# --- monkey-patch the DINOv2 assert (numerics unchanged) ---
enc = model.encoder.model
_orig_interp = enc.interpolate_pos_encoding


def _interp(x, w, h):
    import math
    import torch.nn as nn
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = enc.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return enc.pos_embed
    pos_embed = enc.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // enc.patch_size
    h0 = h // enc.patch_size
    M = int(math.sqrt(N))
    sx = float(w0 + enc.interpolate_offset) / M
    sy = float(h0 + enc.interpolate_offset) / M
    out = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        scale_factor=(sx, sy), mode="bicubic", antialias=enc.interpolate_antialias)
    out = out.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), out), dim=1).to(previous_dtype)


enc.interpolate_pos_encoding = _interp

# --- hooks ---
enc.blocks[23].register_forward_hook(
    lambda m, i, o: save("enc_blk23", o))                       # (1,T,C) T=1+P

model.info_sharing.self_attention_blocks[0].register_forward_pre_hook(
    lambda m, i: save("is_blk0_in", i[0]))                    # (1,T,C) assembled
for idx in (0, 1, 7, 11, 15):
    model.info_sharing.self_attention_blocks[idx].register_forward_hook(
        lambda m, i, o, t=idx: save(f"is_blk{t}", o))           # (1,T,C)

ip = model.dense_head[0].input_process
for idx in range(4):
    ip[idx].register_forward_hook(
        lambda m, i, o, t=idx: save(f"tap{t}", o))              # (1,C,H',W')

for idx, nm in ((4, "ref4"), (3, "ref3"), (2, "ref2"), (1, "ref1")):
    getattr(model.dense_head[0].scratch, f"refinenet{idx}").register_forward_hook(
        lambda m, i, o, t=nm: save(f"dpt_{t}", o))              # (1,256,H',W')

ph = model.pose_head
ph.proj.register_forward_hook(lambda m, i, o: save("ph_proj", o))       # (2,784,H',W')
ph.res_conv[0].register_forward_hook(lambda m, i, o: save("ph_rc0", o))
ph.res_conv[1].register_forward_hook(lambda m, i, o: save("ph_rc1", o))
ph.avgpool.register_forward_hook(lambda m, i, o: save("ph_pool", o))    # (2,784,1,1)
ph.more_mlps.register_forward_hook(lambda m, i, o: save("ph_mlp", o))
model.dense_head[1].register_forward_hook(
    lambda m, i, o: save("reg_out", o.decoded_channels))        # (1,6,H,W)
model.pose_head.register_forward_hook(
    lambda m, i, o: save("pose_raw", o.decoded_channels))       # (1,7)
model.scale_head.register_forward_hook(
    lambda m, i, o: save("scale_raw", o.decoded_channels))      # (1,1,1)

# --- run (B=1, V=S) ---
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
imgs = torch.from_numpy(frames).unsqueeze(0).to(DEV)             # (1,S,3,H,W)
imgs_n = (imgs - mean) / std
views = [{"img": imgs_n[:, v], "data_norm_type": ["dinov2"]}
         for v in range(S)]
with torch.no_grad():
    res = model(views)

# final outputs per view (physical layout)
for v in range(S):
    save(f"out_rays{v}", res[v]["ray_directions"][0])           # (H,W,3)
    save(f"out_depth{v}", res[v]["depth_along_ray"][0])         # (H,W,1)
    save(f"out_conf{v}", res[v]["conf"][0])                     # (H,W,1)
    save(f"out_pts3d{v}", res[v]["pts3d"][0])                   # (H,W,3)
    save(f"out_pose{v}", torch.cat(
        [res[v]["cam_trans"][0], res[v]["cam_quats"][0]], dim=-1))  # (7,)
save("out_scale", res[0]["metric_scaling_factor"][0])           # (1,)
(outdir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
print(f"wrote {len(manifest)} dumps to {outdir}")
