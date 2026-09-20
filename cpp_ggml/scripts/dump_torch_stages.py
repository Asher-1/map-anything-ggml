#!/usr/bin/env python3
"""Dump PyTorch intermediate stages for parity vs the C++ stage dumps."""
import sys
from pathlib import Path
import numpy as np
import torch

REPO_SRC = Path(__file__).resolve().parent.parent / "third_party" / "vggt-omega-src"
sys.path.insert(0, str(REPO_SRC))
from vggt_omega.models import VGGTOmega  # noqa: E402

ckpt, frames_bin, H, W, S, out_dir = sys.argv[1:7]
H, W, S = int(H), int(W), int(S)
Path(out_dir).mkdir(parents=True, exist_ok=True)

frames = np.fromfile(frames_bin, dtype=np.float32).reshape(S, 3, H, W)
images = torch.from_numpy(frames).unsqueeze(0)

model = VGGTOmega(autocast=False)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = sd.get("model", sd.get("state_dict", sd))
# the 256_text checkpoint ships text_alignment_head.* weights: build with
# enable_alignment so the C++ text branch gets a live reference instead of
# dropping the keys (the C++ graph builds the head when the GGUF flag is set)
has_text = any(k.startswith("text_alignment_head.") for k in sd)
if has_text:
    model = VGGTOmega(autocast=False, enable_alignment=True)
    model.load_state_dict(sd, strict=True)
else:
    model.load_state_dict(sd, strict=True)
model.eval()

agg = model.aggregator
with torch.no_grad():
    x = (images - agg._resnet_mean) / agg._resnet_std
    B, S_, C_, H_, W_ = x.shape
    x = x.view(B * S_, C_, H_, W_)
    dino = agg.patch_embed
    with torch.no_grad():
        conv4 = dino.patch_embed.proj(x)  # (B*S, C, Hp, Wp)
        conv4.contiguous().numpy().tofile(f"{out_dir}/torch_conv4.bin")
        tokens0 = dino.prepare_tokens_with_masks(x)  # returns (tokens, hw)
        tokens = tokens0[0] if isinstance(tokens0, tuple) else tokens0
        tokens.contiguous().numpy().tofile(f"{out_dir}/torch_tokens0.bin")
        Hg, Wg = H_ // agg.patch_size, W_ // agg.patch_size
        sin, cos = dino.rope_embed(H=Hg, W=Wg)
        rope = (sin.to(torch.float32), cos.to(torch.float32))
        ident_cos = torch.ones(Hg * Wg + 5, 64)
        ident_sin = torch.zeros(Hg * Wg + 5, 64)
        ident_cos[5:].copy_(cos); ident_sin[5:].copy_(sin)
        ident_cos.numpy().tofile(f"{out_dir}/torch_rope_cos_bb.bin")
        ident_sin.numpy().tofile(f"{out_dir}/torch_rope_sin_bb.bin")
        blk0 = dino.blocks[0]
        with torch.no_grad():
            n1 = blk0.norm1(tokens)
            n1.contiguous().numpy().tofile(f"{out_dir}/torch_b0_norm1.bin")
            qkv = blk0.attn.qkv(n1)
            qkv.contiguous().numpy().tofile(f"{out_dir}/torch_b0_qkv.bin")
            B, N, _ = qkv.shape
            qkv_flat = qkv
            qkv = qkv.reshape(B, N, 3, 16, 64)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            q_r, k_r = blk0.attn.apply_rope(q, k, rope)
            q_r.contiguous().numpy().tofile(f"{out_dir}/torch_b0_qrope.bin")
            v_t = qkv.reshape(B, N, 3, 16, 64).unbind(2)[2].transpose(1, 2)
            v_t.contiguous().numpy().tofile(f"{out_dir}/torch_b0_vperm.bin")
            attn_core = blk0.attn.compute_attention(qkv=qkv_flat, rope=rope)
            attn_core.contiguous().numpy().tofile(f"{out_dir}/torch_b0_fattn.bin")
            proj_o = blk0.attn.proj(attn_core)
            proj_o.contiguous().numpy().tofile(f"{out_dir}/torch_b0_proj.bin")
            h = blk0.ls1(proj_o)
            mid = tokens + h
            mid.numpy().tofile(f"{out_dir}/torch_b0_mid.bin")
            f = blk0.mlp(blk0.norm2(mid))
            f = blk0.ls2(f)
            tokens = mid + f
        tokens.numpy().tofile(f"{out_dir}/torch_blk0.bin")
        tokens = dino.blocks[1](tokens, rope)
        tokens.numpy().tofile(f"{out_dir}/torch_blk1.bin")
        for _i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23):
            tokens = dino.blocks[_i](tokens, rope)
            if _i in (3, 7, 11, 15, 19, 23):
                tokens.contiguous().numpy().tofile(f"{out_dir}/torch_blk{_i}.bin")
        pt = dino.forward_features(x)["x_norm_patchtokens"]
        pt.numpy().tofile(f"{out_dir}/torch_patches.bin")

    with torch.no_grad():
        outputs, pstart = agg(images)
    for li in (4, 11, 17, 23):
        t = outputs[li]  # (B, S, N, 2C)
        # -> ggml {2C, N, S}: (B,S,N,C) -> (S, C, N) [B=1]
        g = t[0].permute(2, 0, 1).contiguous()  # (2C, N, S)
        g.numpy().tofile(f"{out_dir}/torch_inter{li}.bin")

    with torch.no_grad():
        preds = model(images)
ref_extra = {}
if has_text:
    ref_extra["text_embedding"] = preds["text_alignment_embedding"][0].float().numpy().reshape(2048)
np.savez(f"{out_dir}/ref.npz",
         pose_enc=preds["pose_enc"].float().numpy().reshape(S, 9),
         depth=preds["depth"].float().numpy().reshape(S, H, W),
         depth_conf=preds["depth_conf"].float().numpy().reshape(S, H, W),
         **ref_extra)
print("torch stages dumped")

# extra mlp stage dumps (block0) for gelu domain debugging
with torch.no_grad():
    n2 = blk0.norm2(mid)
    n2.contiguous().numpy().tofile(f"{out_dir}/torch_b0_n2.bin")
    f1 = blk0.mlp.fc1(n2)
    f1.contiguous().numpy().tofile(f"{out_dir}/torch_b0_f1.bin")
    import torch.nn.functional as Fg
    ge = Fg.gelu(f1, approximate="none")
    ge.contiguous().numpy().tofile(f"{out_dir}/torch_b0_ge.bin")
print("mlp stages dumped")
