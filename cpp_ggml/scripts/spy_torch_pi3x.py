#!/usr/bin/env python3
"""Dump Pi3X final outputs (f32) for parity against the C++ pi3x builder.

Usage (PYTHONPATH=/tmp/torch_cuda_lib:<repo>):
  python3 scripts/spy_torch_pi3x.py <frames.bin> <H> <W> <S> <out_prefix>

Writes <prefix>.pose16.bin (S,16 camera-to-world row-major, translations
ALREADY in metric scale, matching the official forward), .local_points.bin
(S,H,W,3, metric), .conf.bin (S,H,W), .points.bin (S,H,W,3, metric) and
.scale.bin (metric = exp(raw)).
"""
import sys
from pathlib import Path

import numpy as np
import torch

CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "third_party" / "pi3-src"))
from pi3.models.pi3x import Pi3X  # noqa: E402

frames_bin, H, W, S, out_prefix = (sys.argv[1], int(sys.argv[2]),
                                   int(sys.argv[3]), int(sys.argv[4]),
                                   sys.argv[5])
frames = np.fromfile(frames_bin, dtype=np.float32).reshape(S, 3, H, W)
model = Pi3X.from_pretrained(str(CPP / "models/pytorch/pi3x")).eval()
imgs = torch.from_numpy(frames).unsqueeze(0).float()

# stage dumps (token-major (B*N, T, C) per stage, f32)
import os
dump_dir = os.environ.get("PI3X_DUMP")
if dump_dir:
    def save(name, t):
        a = t.detach().float().cpu().numpy()
        a.tofile(f"{dump_dir}/{name}.bin")
        print("  dump", name, a.shape)
    m0_upsample_blocks = model.point_head.upsample_blocks
    model.encoder.register_forward_hook(
        lambda m, i, o: save("torch_enc_patch",
                             o["x_norm_patchtokens"]))  # (B*N,P,C)
    for idx in (0, 1, 2, 3, 4, 34, 35):
        model.decoder[idx].register_forward_hook(
            lambda m, i, o, t=idx: save(f"torch_dec{t}", o))
    for _li, _blk in enumerate(m0_upsample_blocks):
        # forward uses checkpoint per-LAYER (Sequential.forward never runs),
        # so hook the individual layers: [0][1]=Conv3x3 after ConvT,
        # [1]/[2]=ResidualConvBlocks
        _blk[0][1].register_forward_hook(
            lambda m, i, o, t=_li: save(f"torch_ph_c3{t}", o))
        _blk[1].register_forward_hook(
            lambda m, i, o, t=_li: save(f"torch_ph_res1_{t}", o))
        _blk[-1].register_forward_hook(
            lambda m, i, o, t=_li: save(f"torch_ph_lvl{t}", o))

    # metric branch internals (point/camera decoders rope first on the
    # SHARED RoPE2D; metric q is uniquely identifiable by its N==1 shape)
    mt = model.metric_decoder
    mt.projects_x.register_forward_hook(lambda m, i, o: save("torch_mx0", o))
    mt.projects_y.register_forward_hook(lambda m, i, o: save("torch_my0", o))
    _mcount = [0]
    def _mrope_pre(mod, i):
        if i[0].shape[-2] == 1:
            _mcount[0] += 1
            save(f"torch_m_q_norope{_mcount[0]}", i[0])
    def _mrope_post(mod, i, o):
        if i[0].shape[-2] == 1:
            save(f"torch_m_q_rope{_mcount[0]}", o)
    mt.blocks[0].cross_attn.rope.register_forward_pre_hook(_mrope_pre)
    mt.blocks[0].cross_attn.rope.register_forward_hook(_mrope_post)
    mt.blocks[0].cross_attn.proj.register_forward_hook(
        lambda m, i, o: save("torch_m_sdpa0", o))
    for _i, _blk in enumerate(mt.blocks):
        _blk.register_forward_hook(
            lambda m, i, o, t=_i: save(f"torch_m_blk{t}", o))

    model.point_decoder.register_forward_hook(
        lambda m, i, o: save("torch_pdec_out", o))
    model.conf_decoder.register_forward_hook(
        lambda m, i, o: save("torch_cdec_out", o))
    model.camera_decoder.register_forward_hook(
        lambda m, i, o: save("torch_mdec_out", o))
    model.metric_decoder.register_forward_hook(
        lambda m, i, o: save("torch_mtdec_out", o))

with torch.no_grad():
    res = model(imgs)

pose = res["camera_poses"][0].float().cpu().numpy().reshape(S, 16)
pose.tofile(f"{out_prefix}.pose16.bin")
lp = res["local_points"][0].float().cpu().numpy()
lp.tofile(f"{out_prefix}.local_points.bin")
conf = res["conf"][0].float().cpu().numpy()[..., 0]
conf.tofile(f"{out_prefix}.conf.bin")
pts = res["points"][0].float().cpu().numpy()
pts.tofile(f"{out_prefix}.points.bin")
sc = res["metric"].float().cpu().numpy().reshape(1)
sc.tofile(f"{out_prefix}.scale.bin")
print(f"torch pi3x: metric {sc[0]:.6f}  local |.| mean "
      f"{np.abs(lp).mean():.4f}  wrote outputs to {out_prefix}.*")
