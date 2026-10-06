#!/usr/bin/env python3
"""Convert the original VGGT-1B checkpoint (facebook/VGGT-1B, model.pt) to
GGUF for the mapggml C++ runtime.

Key mapping (torch -> mapggml GGUF), mirroring the vggt-omega converter:
  aggregator.patch_embed.*   -> bb.*        (the DINOv2 ViT-L "backbone")
  aggregator.frame_blocks.N  -> agg.frame.N (within-frame blocks, plain qkv bias)
  aggregator.global_blocks.N -> agg.inter.N (cross-frame blocks, qk-norm + layerscale)
  aggregator.camera_token    -> agg.cam_token
  aggregator.register_token  -> agg.reg_token
  camera_head.*              -> cam.*
  depth_head.*               -> dp.*
  point_head.*               -> pt.*
  track_head.*               -> dropped (deferred; counted and reported)

Architecture metadata written: general.architecture="vggt",
patch_size=14, num_register_tokens=4, depth/aa_depth=24, and the ImageNet
normalization constants the official aggregator applies internally
(_RESNET_MEAN/_RESNET_STD are non-persistent buffers, i.e. constants).

Usage:
  python3 scripts/convert_vggt_to_gguf.py models/pytorch/vggt1b/model.pt \
      models/gguf/vggt-1b-f16.gguf --outtype f16
"""
import argparse
import ctypes
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType

QTYPES = {
    "q8_0": (GGMLQuantizationType.Q8_0, 32, 34),
    "q6_K": (GGMLQuantizationType.Q6_K, 256, 210),
    "q5_K": (GGMLQuantizationType.Q5_K, 256, 176),
}

FILE_TYPE = {
    "f32": 0, "f16": 1, "q8_0": 8, "q6_K": 15, "q5_K": 13,
}

# 2-D Linear weights get quantized; everything else stays at the outtype
# float. Covers the backbone/global-inter blocks, both DPT heads, and the
# camera-head linears.
_LINEAR_PAT = re.compile(
    r"\.(attn\.qkv|attn\.q|attn\.k|attn\.v|attn\.proj|mlp\.fc1|mlp\.fc2|projects\.\d+|projects|"
    r"resize_layers\.\d+|scratch\.layer[1-4]_rn|scratch\.output_conv|"
    r"embed_pose|pose_branch\.fc1|pose_branch\.fc2)\.weight$")


def is_linear_weight(key: str) -> bool:
    return bool(_LINEAR_PAT.search(key))


def unwrap_checkpoint(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def rename_key(k: str) -> str:
    if k.startswith("aggregator.patch_embed."):
        return "bb." + k[len("aggregator.patch_embed."):]
    if k.startswith("aggregator.frame_blocks."):
        rest = k[len("aggregator.frame_blocks."):]
        idx, tail = rest.split(".", 1)
        return f"agg.frame.{idx}.{tail}"
    if k.startswith("aggregator.global_blocks."):
        rest = k[len("aggregator.global_blocks."):]
        idx, tail = rest.split(".", 1)
        return f"agg.inter.{idx}.{tail}"
    if k == "aggregator.camera_token":
        return "agg.cam_token"
    if k == "aggregator.register_token":
        return "agg.reg_token"
    if k.startswith("camera_head."):
        return "cam." + k[len("camera_head."):]
    if k.startswith("depth_head."):
        return "dp." + k[len("depth_head."):]
    if k.startswith("point_head."):
        return "pt." + k[len("point_head."):]
    return k


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--outtype", default="f16",
                    choices=["f32", "f16", "q8_0", "q6_K", "q5_K"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.outtype in ("q8_0", "q6_K", "q5_K"):
        ctypes_lib = load_ggml_lib()
    else:
        ctypes_lib = None

    print(f"loading {args.checkpoint} ...")
    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = unwrap_checkpoint(obj)
    assert isinstance(sd, dict) and sd, "unexpected checkpoint format"

    # Map names; split fused qkv (the C++ attention consumes separate q/k/v);
    # count dropped groups.
    mapped, dropped = {}, []
    for k, t in sd.items():
        if k.startswith("track_head."):
            dropped.append(k)
            continue
        nk = rename_key(k)
        assert nk != k or k.startswith(("bb.", "agg.", "cam.", "dp.", "pt.")), \
            f"unmapped key: {k}"
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
    print(f"tensors: {len(mapped)} mapped, {len(dropped)} track_head dropped")

    # Architecture facts (from the official VGGT aggregator/vggt.py defaults).
    n_layers = len({k.split(".")[2] for k in mapped
                    if k.startswith("bb.blocks.")}) if any(
        k.startswith("bb.blocks.") for k in mapped) else 24
    meta = {
        "general.name": "vggt-1b",
        "general.architecture": "vggt",
        "vggt.dtype": args.outtype,
        "vggt.patch_size": 14,
        "vggt.embed_dim": 1024,
        "vggt.depth": n_layers,
        "vggt.aa_depth": 24,
        "vggt.num_heads": 16,
        "vggt.num_register_tokens": 4,
        "vggt.trunk_depth": 4,
        "vggt.dpt_features": 1024,   # VGGT DPT scratch channels
        "vggt.cached_layer_idx": [4, 11, 17, 23],
        # The official aggregator normalizes with the ImageNet constants
        # (non-persistent buffers): _RESNET_MEAN/_RESNET_STD.
        "vggt.img_mean": [0.485, 0.456, 0.406],
        "vggt.img_std": [0.229, 0.224, 0.225],
        # VGGT-specific notes for the builder (documented, consumed later):
        "vggt.rope_mode": "rotary2d_f100",   # RotaryPositionEmbedding2D(100)
        "vggt.patch_embed_mode": "dinov2",   # pos_embed additive, no rope
    }

    if args.dry_run:
        for k, v in meta.items():
            print(f"  {k} = {v}")
        print(f"would write {len(mapped)} tensors to {args.output}")
        return

    # Quantization: only 2-D Linear weights (q8_0 via gguf-py, K-quants via
    # ctypes ggml_quantize_chunk); everything else keeps the outtype float.
    out = GGUFWriter(args.output, "vggt")
    for k, v in meta.items():
        if isinstance(v, list):
            # this gguf-py version's _pack_val only accepts python sequences
            out.add_array(k, list(v))
        elif isinstance(v, int):
            out.add_uint32(k, v)
        else:
            out.add_string(k, str(v))
    out.add_uint32("general.file_type", FILE_TYPE[args.outtype])

    def coerce(t: torch.Tensor, name: str):
        # omega rule: only 2-D Linear weights follow the outtype; every other
        # tensor (biases, norms, layer scales, token params, pos_embed, rope
        # tables) stays f32 — F16 token params would break type-mixed concat
        # and shift the numeric baseline.
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
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(mapped)}")

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    size = Path(args.output).stat().st_size / 1e9
    print(f"wrote {args.output} ({size:.2f} GB, {len(mapped)} tensors)")


if __name__ == "__main__":
    main()
