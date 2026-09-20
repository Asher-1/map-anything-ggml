#!/usr/bin/env python3
"""Dump pi3 stage tensors for parity debugging against the C++ builder.

Usage (PYTHONPATH=/tmp/torch_cuda_lib:<pi3-src>:<repo>):
  python3 scripts/spy_torch_pi3.py <frames.bin> <H> <W> <S> <out_dir>

All stage dumps are f32 token-major (S, T, C) so a C++ dump {C, T, S}
(ne[0]=C fastest) reshapes to the identical memory layout.  Exceptions
are stated per file.  Also writes the raw pos_embed / cls_token /
register_tokens params and an md5 manifest for freshness checks.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # CPU-torch
# fallbacks (plain python3 envs) must still produce a valid reference


CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "third_party" / "pi3-src"))
from pi3.models.pi3 import Pi3  # noqa: E402

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
model = Pi3.from_pretrained(str(CPP / "models/pytorch/pi3")).eval()
model = model.to(DEV).float()

enc = model.encoder

# ---- hooks (inputs are (B*N, ...) with B=1, so leading dim == S) ----
enc.patch_embed.register_forward_pre_hook(
    lambda m, inp: save("normed_img", inp[0]))            # (S,3,H,W)
enc.patch_embed.proj.register_forward_hook(
    lambda m, i, o: save("conv_out", o))                  # (S,C,Hp,Wp) conv layout

_orig_interp = enc.interpolate_pos_encoding


def _interp(x, w, h):
    out = _orig_interp(x, w, h)
    save("pos_interp", out)                               # (S,1+P,C)
    return out


enc.interpolate_pos_encoding = _interp

enc.blocks[0].register_forward_pre_hook(
    lambda m, inp: save("blk0_in", inp[0]))               # (S,1+4+P,C)
enc.register_forward_hook(
    lambda m, i, o: save("enc_patch", o["x_norm_patchtokens"]))  # (S,P,C)

def _dec_hook(idx):
    return lambda m, i, o: save(f"dec{idx}_out", o)
for _i in (0, 1, 2, 3, 4):
    model.decoder[_i].register_forward_hook(_dec_hook(_i))

_orig_decode = model.decode


def _decode(hidden, N, H_, W_):
    hid, pos = _orig_decode(hidden, N, H_, W_)
    save("hidden2", hid)                                  # (S,5+P,2048)
    return hid, pos


model.decode = _decode

for tag, mod in (("pd", model.point_decoder), ("cd", model.conf_decoder),
                 ("md", model.camera_decoder)):
    mod.projects.register_forward_hook(
        lambda m, i, o, t=tag: save(f"{t}_proj", o))      # (S,5+P,C)
    mod.blocks[0].register_forward_hook(
        lambda m, i, o, t=tag: save(f"{t}_blk0", o))
    mod.blocks[4].register_forward_hook(
        lambda m, i, o, t=tag: save(f"{t}_blk4", o))

model.point_decoder.register_forward_hook(
    lambda m, i, o: save("pdec_out", o))                  # (S,5+P,C)
model.conf_decoder.register_forward_hook(
    lambda m, i, o: save("cdec_out", o))                  # (S,5+P,C)
model.point_head.proj.register_forward_hook(
    lambda m, i, o: save("phead_lin", o))                 # (S,3*196,P)
model.conf_head.proj.register_forward_hook(
    lambda m, i, o: save("chead_lin", o))                 # (S,1*196,P)
model.point_head.register_forward_hook(
    lambda m, i, o: save("phead_out", o))                 # (S,H,W,3) raw
model.conf_head.register_forward_hook(
    lambda m, i, o: save("chead_out", o))                 # (S,H,W,1) raw

# ---- raw params for GGUF cross-checks ----
save("p_pos_embed", model.encoder.pos_embed[0])           # (1+P0,C)
save("p_cls_token", model.encoder.cls_token[0, 0])        # (C,)
save("p_reg_tokens", model.encoder.register_tokens[0])    # (4,C)
save("p_dec_reg", model.register_token[0, 0])             # (5,C)

images = torch.from_numpy(frames).unsqueeze(0).to(DEV)
with torch.no_grad():
    preds = model(images)

save("local_points", preds["local_points"][0])            # (S,H,W,3)
save("conf", preds["conf"][0])                            # (S,H,W,1)
save("pose16", preds["camera_poses"][0].reshape(S, 16))
(outdir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
print(f"wrote {len(manifest)} dumps to {outdir}")
