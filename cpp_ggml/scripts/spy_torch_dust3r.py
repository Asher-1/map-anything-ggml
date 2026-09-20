#!/usr/bin/env python3
"""Dump DUSt3R stage tensors for parity debugging against the C++ builder.

Usage:
  PYTHONPATH=<cpp_ggml>/third_party/dust3r-src \
    python3 scripts/spy_torch_dust3r.py <frames.bin> <H> <W> <S=2> <out_dir>

frames.bin is S*3*H*W f32 in [0,1] (torch (S,3,H,W) order — the CLI --bin
format); the official ImgNorm ((x-0.5)/0.5) is applied HERE, matching the
C++ graph's mean/std transform.

All stage dumps are f32 token-major (S, T, C) so a C++ dump {C, T, S}
(ne[0]=C fastest) reshapes to the identical memory layout; conv-layout
tensors keep torch (B, C, Hp, W) — stated per file.  Writes an md5
manifest for freshness checks.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "third_party" / "dust3r-src"))
import dust3r.utils.path_to_croco  # noqa: F401,E402  (adds croco to sys.path)
from dust3r.model import AsymmetricCroCo3DStereo  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

frames_bin, H, W, S, out_dir = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                int(sys.argv[4]), sys.argv[5])
assert S == 2, "dust3r is pair-wise: S must be 2"
outdir = Path(out_dir)
outdir.mkdir(parents=True, exist_ok=True)

manifest = []


def save(name, arr):
    arr = np.ascontiguousarray(arr.detach().float().cpu().numpy())
    arr.tofile(outdir / f"{name}.bin")
    md5 = hashlib.md5((outdir / f"{name}.bin").read_bytes()).hexdigest()[:10]
    manifest.append(f"{name} {arr.shape} {md5}")
    print(f"  {name} {arr.shape} {md5}", flush=True)


# HF PyTorchModelHubMixin layout: config.json + model.safetensors in the
# local dir (the repo ships no .pth). landscape_only stays whatever the
# config says — for landscape (W >= H) batches the wrapper is a pass-through
# (misc.py wrapper_yes: is_landscape.all() -> head(decout, (H, W))).
model = AsymmetricCroCo3DStereo.from_pretrained(
    str(CPP / "models/pytorch/dust3r")).float().eval().to(DEV)

frames = np.fromfile(frames_bin, dtype=np.float32).reshape(S, 3, H, W)
# official ImgNorm: (x - 0.5) / 0.5
imgs = torch.from_numpy(frames).to(DEV) * 2.0 - 1.0   # (S,3,H,W), normalized
true_shape = torch.tensor([[H, W]] * S, dtype=torch.int64, device=DEV)

# ---- hooks (the pair is batched: encoder tensors have B=2) ----
model.patch_embed.proj.register_forward_hook(
    lambda m, i, o: save("conv_out", o))                  # (2,C,Hp,Wp)
model.enc_blocks[0].register_forward_hook(
    lambda m, i, o: save("enc_blk0_out", o))              # (2,P,1024)
model.enc_norm.register_forward_hook(
    lambda m, i, o: save("enc_out", o))                   # (2,P,1024)
_decproj_calls = []


def _decproj_hook(m, i, o):
    # decoder_embed fires twice (f1 then f2 — model.py _decoder)
    _decproj_calls.append(1)
    save(f"dec_proj_f{len(_decproj_calls)}", o)           # (1,P,768) each


model.decoder_embed.register_forward_hook(_decproj_hook)


def _dec1_hook(idx):
    return lambda m, i, o: save(f"dec1_blk{idx}_out", o[0])  # (1,P,768)


def _dec2_hook(idx):
    return lambda m, i, o: save(f"dec2_blk{idx}_out", o[0])  # (1,P,768)


def _b0_attn_pre(m, inp):
    # inp[0] = norm1(x) (1,P,768); rope happens inside — recompute torch
    # rope on q/k here for layout comparison
    pass
model.dec_blocks[0].attn.register_forward_pre_hook(_b0_attn_pre)
_orig_b0_attn = model.dec_blocks[0].attn.forward
def _b0_attn_fwd(x, xpos):
    B, N, C = x.shape
    qkv = model.dec_blocks[0].attn.qkv(x).reshape(B, N, 3, 12, 64).transpose(1, 3)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    rope = model.rope
    qr = rope(q, xpos); kr = rope(k, xpos)
    save("b0_qrope_torch", qr[0].permute(1, 0, 2).reshape(N, 768))
    save("b0_krope_torch", kr[0].permute(1, 0, 2).reshape(N, 768))
    save("b0_qraw_torch", q[0].permute(1, 0, 2).reshape(N, 768))
    return _orig_b0_attn(x, xpos)
model.dec_blocks[0].attn.forward = _b0_attn_fwd
model.dec_blocks[0].attn.register_forward_hook(
    lambda m, i, o: save("b0_selfattn", o))               # (1,P,768)
model.dec_blocks[0].cross_attn.register_forward_hook(
    lambda m, i, o: save("b0_cross", o))                  # (1,P,768)
model.dec_blocks[0].mlp.register_forward_hook(
    lambda m, i, o: save("b0_mlp", o))                    # (1,P,768)
for _i in (0, 1, 5, 8):
    model.dec_blocks[_i].register_forward_hook(_dec1_hook(_i))
    model.dec_blocks2[_i].register_forward_hook(_dec2_hook(_i))
# dec_norm fires twice per forward (view1 side first, then view2 —
# model.py: tuple(map(self.dec_norm, final_output[-1]))); disambiguate
# by call order.
_decnorm_calls = []


def _decnorm_hook(m, i, o):
    _decnorm_calls.append(1)
    save("dec1_final" if len(_decnorm_calls) == 1 else "dec2_final", o)


model.dec_norm.register_forward_hook(_decnorm_hook)

# DPT raw output (pre-postprocess) per head: (1, 4, H, W) — layout matches
# the C++ out_raw {W, H, 4, 1} (w fastest, then h, then channel).
model.downstream_head1.dpt.register_forward_hook(
    lambda m, i, o: save("head1_dpt_raw", o))
model.downstream_head2.dpt.register_forward_hook(
    lambda m, i, o: save("head2_dpt_raw", o))

view1 = dict(img=imgs[0:1], true_shape=true_shape[0:1], instance="v1")
view2 = dict(img=imgs[1:2], true_shape=true_shape[1:2], instance="v2")
with torch.no_grad():
    res1, res2 = model(view1, view2)

save("pts3d_view1", res1["pts3d"][0])                     # (H,W,3)
save("conf_view1", res1["conf"][0])                       # (H,W,1)
save("pts3d_view2_in_view1", res2["pts3d_in_other_view"][0])  # (H,W,3)
save("conf_view2", res2["conf"][0])                       # (H,W,1)
(outdir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
print(f"wrote {len(manifest)} dumps to {outdir}")
