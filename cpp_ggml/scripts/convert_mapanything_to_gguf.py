#!/usr/bin/env python3
"""Convert the MapAnything checkpoint (facebook/map-anything, model.safetensors)
to GGUF for the mapggml C++ runtime.

Key mapping (torch -> mapggml GGUF):
  encoder.model.*             -> bb.*      (DINOv2 ViT-G/14 backbone, first 24
                                             blocks, swiglu MLP, NO final norm;
                                             the bb.* prefix reuses the shared
                                             encoder path)
  encoder.model.blocks.N.attn.qkv  -> bb.blocks.N.attn.{q,k,v}.* (fused split)
  info_sharing.self_attention_blocks.N.* -> is.N.*  (16 AAT blocks, same block
                                             structure: swiglu, ls 1e-5, no qk
                                             norm, no rope)
  info_sharing.norm.*         -> is_norm.* (final LN, also applied to IFR[7,11])
  info_sharing.view_pos_table -> is_view_pe (sin/cos buffer, ref view row 0)
  scale_token                 -> is_scale_token
  dense_head.0.*              -> dp.*   (DPTFeature: input_process chain,
                                         scratch.refinenet1..4)
  dense_head.1.*              -> dr.*   (DPTRegressionProcessor: conv1, conv2.0/2)
  pose_head.*                 -> cam.*  (proj + 2 ResConv + more_mlps + fc_t/fc_rot(4))
  scale_head.*                -> scl.*  (MLPHead: proj + 2x[lin+relu] + output_proj)
  DROPPED: geometric-input encoders (cam_rot/cam_trans/depth/ray_dirs/
  scale/depth_scale) — only used when geometric inputs are provided.  The
  fusion_norm_layer is KEPT: the official forward applies it to the encoder
  features UNCONDITIONALLY, even with images_only inputs.

Architecture metadata: general.architecture="mapanything", patch 14,
dim 1536, heads 24, enc_depth 24, is_depth 16, AAT swiglu hidden 4096,
ifr_indices [7,11] (DPT consumes encoder final + AAT IFR[0,1] + final),
pose rot dim 4 (quaternion), ImageNet image normalization.

Usage:
  python3 scripts/convert_mapanything_to_gguf.py \
      models/pytorch/mapanything/model.safetensors \
      models/gguf/mapanything-f16.gguf --outtype f16
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
from safetensors.torch import load_file

QTYPES = {
    "q8_0": (GGMLQuantizationType.Q8_0, 32, 34),
    "q5_K": (GGMLQuantizationType.Q5_K, 256, 176),
}

FILE_TYPE = {
    "f32": 0, "f16": 1, "q8_0": 8, "q5_K": 13,
}

# 2-D Linear weights get quantized; everything else stays at the outtype
# float (biases, norms, layer scales, token params, pos_embed, view_pe).
_LINEAR_PAT = re.compile(
    r"\.(attn\.(qkv|q|k|v|proj)|mlp\.(w12|w3)|fc_t|fc_rot|more_mlps\.[02]|"
    r"output_proj|proj|mlp\.[01]\.0)\.weight$"
)


def is_linear_weight(key: str) -> bool:
    return bool(_LINEAR_PAT.search(key))


def rename_key(k: str) -> str | None:
    """Map a checkpoint key to the GGUF key; None = drop."""
    if k.startswith("encoder.model."):
        return "bb." + k[len("encoder.model."):]
    if k.startswith("info_sharing.self_attention_blocks."):
        return "is." + k[len("info_sharing.self_attention_blocks."):]
    if k.startswith("info_sharing."):
        return "is_" + k[len("info_sharing."):]
    if k == "scale_token":
        return "is_scale_token"
    if k.startswith("dense_head.0."):
        return "dp." + k[len("dense_head.0."):]
    if k.startswith("dense_head.1."):
        return "dr." + k[len("dense_head.1."):]
    if k.startswith("pose_head."):
        return "cam." + k[len("pose_head."):]
    if k.startswith("scale_head."):
        return "scl." + k[len("scale_head."):]
    if k.startswith("fusion_norm_layer."):
        return k  # applied to encoder features even in images_only mode
    return None  # geometric-input encoders: unused


def quantize_k(ctypes_lib, t, qt):
    """Quantize a (out,in) f32 matrix via ggml_quantize_chunk (ctypes)."""
    dtype, blck, tsize = QTYPES[qt]
    # drift guard: cross-check the hard-coded table against the loaded ggml
    # (a wrong tsize makes dst too small -> segfault inside quantize_chunk)
    ts = ctypes_lib.ggml_type_size(int(dtype))
    bs = ctypes_lib.ggml_blck_size(int(dtype))
    assert (bs, ts) == (blck, tsize), (qt, "ggml says", bs, ts, "table says", blck, tsize)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--outtype", default="f16",
                    choices=["f32", "f16", "q8_0", "q5_K"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.outtype in ("q8_0", "q5_K"):
        ctypes_lib = load_ggml_lib()
    else:
        ctypes_lib = None

    print(f"loading {args.checkpoint} ...")
    sd = load_file(args.checkpoint)
    assert isinstance(sd, dict) and sd, "unexpected checkpoint format"

    mapped, dropped = {}, []
    for k, t in sd.items():
        nk = rename_key(k)
        if nk is None:
            dropped.append(k)
            continue
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

    # ImageNet constants (uniception data_norm_type="dinov2")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    meta = {
        "general.name": "mapanything",
        "general.architecture": "mapanything",
        "mapanything.dtype": args.outtype,
        "mapanything.patch_size": 14,
        "mapanything.embed_dim": 1536,
        "mapanything.num_heads": 24,
        "mapanything.enc_depth": count_prefix(mapped, "bb.blocks.", 2),
        "mapanything.is_depth": count_prefix(mapped, "is.", 1),
        "mapanything.swiglu_hidden": 4096,   # SwiGLUFFNFused(1536*4*2/3 -> mult of 8)
        "mapanything.ifr_index0": 7,          # AAT IFR indices; DPT consumes
        "mapanything.ifr_index1": 11,         # enc final + IFR[0,1] + final
        "mapanything.pose_rot_dim": 4,        # quaternion
        "mapanything.img_mean": mean,
        "mapanything.img_std": std,
        "mapanything.notes": "dense 6ch = raydirs(3 unit-sphere) + depth(exp) "
                             "+ conf(1+exp) + mask(sigmoid); pose t(3)+quat(4, "
                             "L2-normalized); scale = exp from the scale token; "
                             "ref-view PE = is_view_pe row 0 added to view 0.",
    }

    if args.dry_run:
        for k, v in meta.items():
            print(f"  {k} = {v}")
        print(f"would write {len(mapped)} tensors to {args.output}")
        return

    out = GGUFWriter(args.output, "mapanything")
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
        # block size; heads with odd output dims (e.g. (4, 816)) stay float
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
    out.close()
    size = Path(args.output).stat().st_size / 1e9
    print(f"wrote {args.output} ({size:.2f} GB, {len(mapped)} tensors)")


if __name__ == "__main__":
    main()
