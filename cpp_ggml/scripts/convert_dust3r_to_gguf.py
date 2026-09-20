#!/usr/bin/env python3
"""Convert the DUSt3R checkpoint (naver DUSt3R_ViTLarge_BaseDecoder_512_dpt)
to GGUF for the cpp_ggml Dust3rImpl builder (M5).

Source layout (dust3r AsymmetricCroCo3DStereo extends croco CroCoNet):
  patch_embed.proj/norm        -> bb_patch.*     (Conv2d 3->1024 k16 s16 + LN)
  enc_blocks.N.*               -> enc.N.*        (24 x Block, fused qkv)
  enc_norm.*                   -> enc_norm.*
  decoder_embed.*              -> dec_embed.*    (Linear 1024->768)
  dec_blocks.N.*               -> dec.0.N.*      (view1-side, 12 x DecoderBlock)
  dec_blocks2.N.*              -> dec.1.N.*      (view2-side, independent weights)
  dec_norm.*                   -> dec_norm.*
  downstream_head1.dpt.*       -> head1.*        (DPT, xyz+conf 4ch)
  downstream_head2.dpt.*       -> head2.*
Dropped: mask_token / mask generator / prediction_head (croco's own head,
unused by dust3r) / pos-embed buffers (RoPE mode has none).

Coercion (omega rule): 2-D Linear weights follow the outtype; every other
tensor (LayerNorms, conv kernels/biases, DPT convs) stays f32 — matching
convert_pi3_to_gguf.py.  Conv2d/ConvTranspose2d kernels are copied verbatim
(the same memcpy the vggt/mapanything converters already validated).

Usage:
  python3 scripts/convert_dust3r_to_gguf.py \
      models/pytorch/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
      models/gguf/dust3r-f16.gguf --outtype f16 [--dry-run]
"""
import argparse
import ctypes
import os
import re
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType

QTYPES = {
    "q8_0": (GGMLQuantizationType.Q8_0, 32, 34),
    "q5_K": (GGMLQuantizationType.Q5_K, 256, 176),
}
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 8, "q5_K": 13}

# 2-D Linear weights follow the outtype; 4-D conv kernels and norms stay f32.
# croco attention: fused qkv (self) + separate projq/projk/projv (cross);
# dust3r decoder_embed is a plain Linear.
_LINEAR_PAT = re.compile(
    r"\.(attn\.(qkv|q|k|v)|attn\.proj|cross_attn\.(projq|projk|projv|proj)|"
    r"mlp\.fc1|mlp\.fc2|decoder_embed)\.weight$")


def is_linear_weight(key: str) -> bool:
    return bool(_LINEAR_PAT.search(key))


def rename_key(k: str) -> str:
    if k.startswith("patch_embed."):
        return "bb_patch." + k[len("patch_embed."):]
    if k.startswith("enc_blocks."):
        return "enc." + k[len("enc_blocks."):]
    if k.startswith("enc_norm."):
        return k
    if k == "decoder_embed.weight" or k == "decoder_embed.bias":
        return k
    if k.startswith("dec_blocks2."):
        return "dec.1." + k[len("dec_blocks2."):]
    if k.startswith("dec_blocks."):
        return "dec.0." + k[len("dec_blocks."):]
    if k.startswith("dec_norm."):
        return k
    if k.startswith("downstream_head1.dpt."):
        return "head1." + k[len("downstream_head1.dpt."):]
    if k.startswith("downstream_head2.dpt."):
        return "head2." + k[len("downstream_head2.dpt."):]
    return k


def quantize_k(ctypes_lib, t, qt):
    """Quantize a (out,in) f32 matrix via ggml_quantize_chunk (ctypes)."""
    dtype, blck, tsize = QTYPES[qt]
    # drift guard: cross-check the hard-coded table against the loaded ggml
    ts = ctypes_lib.ggml_type_size(int(dtype))
    bs = ctypes_lib.ggml_blck_size(int(dtype))
    assert (bs, ts) == (blck, tsize), (qt, "ggml says", bs, ts,
                                       "table says", blck, tsize)
    rows, cols = t.shape
    dst = np.empty(rows * cols // blck * tsize, dtype=np.uint8)
    src = np.ascontiguousarray(t)
    n = ctypes_lib.ggml_quantize_chunk(
        int(dtype),
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.c_void_p),
        0, rows, cols, None)
    assert n == dst.size, (n, dst.size)
    return dst.reshape(rows, cols // blck * tsize), dtype


def find_ggml_lib() -> str:
    root = Path(__file__).resolve().parent.parent
    candidates = []
    if os.environ.get("MAPGGML_BUILD_DIR"):
        candidates.append(Path(os.environ["MAPGGML_BUILD_DIR"]) / "lib")
    candidates += sorted(root.glob("build-*/lib"))
    for d in candidates:
        for so in ("libggml-cpu.so", "libggml-cpu.dylib", "libggml-cpu.dll"):
            p = d / so
            if p.exists():
                return str(p)
    raise FileNotFoundError(
        "libggml-cpu not found; build first (cmake --build cpp_ggml/build-cpu) "
        "or set MAPGGML_BUILD_DIR")


def load_ggml_lib():
    lib = ctypes.CDLL(find_ggml_lib())
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_type_size.restype = ctypes.c_size_t
    lib.ggml_type_size.argtypes = [ctypes.c_int]
    lib.ggml_blck_size.restype = ctypes.c_int
    lib.ggml_blck_size.argtypes = [ctypes.c_int]
    return lib


def count_prefix(sd: dict, prefix: str, level: int) -> int:
    return len({k.split(".")[level] for k in sd
                if k.startswith(prefix) and k.count(".") > level})


def load_ckpt(path: str) -> dict:
    """dust3r checkpoints: raw state_dict, {'model': state_dict} (croco
    training format), or the HF-repo safetensors (PyTorchModelHubMixin
    layout — a plain state_dict)."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
    return {k: v for k, v in sd.items() if hasattr(v, "shape")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--outtype", default="f16",
                    choices=["f32", "f16", "q8_0", "q5_K"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ctypes_lib = load_ggml_lib() if args.outtype in ("q8_0", "q5_K") else None

    print(f"loading {args.checkpoint} ...")
    sd = load_ckpt(args.checkpoint)
    assert isinstance(sd, dict) and sd, "unexpected checkpoint format"

    mapped, dropped = {}, []
    for k, t in sd.items():
        nk = rename_key(k)
        if nk == k and not k.startswith(("enc.", "dec.", "decoder_embed",
                                         "dec_norm", "enc_norm", "bb_patch",
                                         "head1.", "head2.")):
            dropped.append(k)
            continue
        # split fused qkv (self-attention; the cross-attention already has
        # separate projq/projk/projv)
        if nk.endswith(".attn.qkv.weight") and t.ndim == 2:
            rows = t.shape[0] // 3
            base = nk[: -len(".attn.qkv.weight")]
            for j, nm in enumerate(("q", "k", "v")):
                mapped[f"{base}.attn.{nm}.weight"] = t[j * rows:(j + 1) * rows, :]
            continue
        if nk.endswith(".attn.qkv.bias") and t.ndim == 1:
            rows = t.shape[0] // 3
            base = nk[: -len(".attn.qkv.bias")]
            for j, nm in enumerate(("q", "k", "v")):
                mapped[f"{base}.attn.{nm}.bias"] = t[j * rows:(j + 1) * rows]
            continue
        mapped[nk] = t
    print(f"tensors: {len(mapped)} mapped, {len(dropped)} dropped")
    if dropped:
        print("  dropped:", dropped[:10], "..." if len(dropped) > 10 else "")

    meta = {
        "general.name": "dust3r",
        "general.architecture": "dust3r",
        "dust3r.dtype": args.outtype,
        "dust3r.patch_size": 16,
        "dust3r.enc_embed_dim": 1024,
        "dust3r.enc_depth": count_prefix(mapped, "enc.", 1),
        "dust3r.enc_num_heads": 16,
        "dust3r.dec_embed_dim": 768,
        "dust3r.dec_depth": count_prefix(mapped, "dec.0.", 2),
        "dust3r.dec_num_heads": 12,
        "dust3r.rope_freq": 100,
        # dust3r's own normalization: (x - 0.5) / 0.5  (ImgNorm, [-1, 1])
        "dust3r.img_mean": [0.5, 0.5, 0.5],
        "dust3r.img_std": [0.5, 0.5, 0.5],
        "dust3r.notes": "pair-wise model: outputs are pts3d for BOTH views in "
                        "view1's frame (head1 = pts3d_in_self_view, head2 = "
                        "pts3d_in_other_view) + conf = 1+exp(x); the model has "
                        "NO pose head (pose comes from global-alignment "
                        "optimization outside the network).",
    }

    if args.dry_run:
        for k, v in meta.items():
            print(f"  {k} = {v}")
        print(f"would write {len(mapped)} tensors to {args.output}")
        return

    out = GGUFWriter(args.output, "dust3r")
    for k, v in meta.items():
        if isinstance(v, list):
            out.add_array(k, list(v))
        elif isinstance(v, int):
            out.add_uint32(k, v)
        else:
            out.add_string(k, str(v))
    out.add_uint32("general.file_type", FILE_TYPE[args.outtype])

    def coerce(t: torch.Tensor, name: str):
        t = t.detach().float()
        if args.outtype == "f16" and is_linear_weight(name) and t.ndim == 2:
            t = t.half()
        return t.cpu().contiguous().numpy()

    for i, (k, t) in enumerate(mapped.items()):
        # block quantization requires the row length to be a multiple of the
        # block size; otherwise stay float
        if (ctypes_lib is not None and is_linear_weight(k) and t.ndim == 2
                and t.shape[1] % QTYPES[args.outtype][1] == 0):
            q, dtype = quantize_k(ctypes_lib, t.detach().float(), args.outtype)
            out.add_tensor(k, q, raw_dtype=dtype)
        else:
            out.add_tensor(k, coerce(t, k))
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(mapped)}")

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
