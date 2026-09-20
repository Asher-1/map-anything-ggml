#!/usr/bin/env python3
"""Convert the Pi3X checkpoint (yyfz233/Pi3X, model.safetensors) to GGUF.

Key mapping (the pi3x port blueprint; see FEATURE_PARITY_AUDIT.md):
  encoder.*                    -> bb.*      (dinov2_vitl14_reg, identical to pi3)
  register_token               -> dec.reg_token
  decoder.N.*                  -> dec.N.*   (36 alternating BlockRope blocks)
  point_decoder.*              -> pdec.*    (TransformerDecoder)
  point_head.*                 -> phead.*   (ConvHead, dim_out=[2,1])
  conf_decoder.*               -> cdec.*
  conf_head.*                  -> chead.*   (ConvHead, dim_out=[1])
  camera_decoder.*             -> mdec.*
  camera_head.*                -> mhead.*   (raw fc_t/fc_rot(9); host SVD)
  metric_token                 -> mtok
  metric_decoder.*             -> mtdec.*   (ContextOnlyTransformerDecoder)
  metric_head.*                -> mth.*
  image_mean / image_std       -> pi3x.img_mean / pi3x.img_std
  depth_encoder.* / depth_emb / ray_embed / pose_inject_blk.*
                               -> DROPPED (images-only inference; the
                                  disable_multimodal() path)
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
from safetensors.torch import load_file

QTYPES = {
    "q8_0": (GGMLQuantizationType.Q8_0, 32, 34),
    "q5_K": (GGMLQuantizationType.Q5_K, 256, 176),
}
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 8, "q5_K": 13}
_LINEAR_PAT = re.compile(
    r"\.(attn\.(qkv|q|k|v|proj)|cross_attn\.(qkv|q|k|v|proj)|mlp\.(fc1|fc2|w12|w3)|"
    r"fc_t|fc_rot|more_mlps\.[02]|projects_x|projects_y|projects|linear_out|"
    r"output_block\.\d+\.2)\.weight$")


def is_linear_weight(key: str) -> bool:
    return bool(_LINEAR_PAT.search(key))


def rename_key(k: str):
    """-> (gguf_key, keep). Multimodal-conditioning weights are dropped."""
    if k.startswith("depth_encoder."):
        return None, False
    if k in ("depth_emb",):
        return None, False
    if k.startswith("ray_embed."):
        return None, False
    if k.startswith("pose_inject_blk."):
        return None, False
    if k.startswith("encoder."):
        return "bb." + k[len("encoder."):], True
    if k == "register_token":
        return "dec.reg_token", True
    if k == "metric_token":
        return "mtok", True
    if k.startswith("decoder."):
        return "dec." + k[len("decoder."):], True
    if k.startswith("point_decoder."):
        return "pdec." + k[len("point_decoder."):], True
    if k.startswith("point_head."):
        return "phead." + k[len("point_head."):], True
    if k.startswith("conf_decoder."):
        return "cdec." + k[len("conf_decoder."):], True
    if k.startswith("conf_head."):
        return "chead." + k[len("conf_head."):], True
    if k.startswith("camera_decoder."):
        return "mdec." + k[len("camera_decoder."):], True
    if k.startswith("camera_head."):
        return "mhead." + k[len("camera_head."):], True
    if k.startswith("metric_decoder."):
        return "mtdec." + k[len("metric_decoder."):], True
    if k.startswith("metric_head."):
        return "mth." + k[len("metric_head."):], True
    return k, True


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
    raise FileNotFoundError("libggml-cpu not found")


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


def quantize_k(ctypes_lib, t, qt):
    dtype, blck, tsize = QTYPES[qt]
    ts = ctypes_lib.ggml_type_size(int(dtype))
    bs = ctypes_lib.ggml_blck_size(int(dtype))
    assert (bs, ts) == (blck, tsize), (qt, bs, ts, blck, tsize)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--outtype", default="f16",
                    choices=["f32", "f16", "q8_0", "q5_K"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    def coerce(t, k):
        """omega rule: only 2-D Linear weights follow the outtype (f16 ->
        half, q8_0/q5_K -> quantize branch); every other tensor (norms,
        biases, token params, pos_embed, conv kernels) stays f32."""
        t = t.detach().float()
        if args.outtype == "f16" and is_linear_weight(k) and t.ndim == 2:
            t = t.half()
        return t.cpu().contiguous().numpy()

    print(f"loading {args.checkpoint} ...")
    sd = load_file(args.checkpoint)
    assert isinstance(sd, dict) and sd

    ctypes_lib = None
    if args.outtype in ("q8_0", "q5_K"):
        ctypes_lib = load_ggml_lib()

    mapped, dropped = {}, []
    for k, t in sd.items():
        nk, keep = rename_key(k)
        if not keep:
            dropped.append(k)
            continue
        # split fused qkv (the C++ attention consumes separate q/k/v)
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
    print(f"{len(sd)} tensors -> {len(mapped)} kept, {len(dropped)} dropped")
    if dropped:
        print("  dropped:", sorted(dropped)[:6], "...")

    # architecture facts for the loader
    emb = mapped["bb.patch_embed.proj.weight"].shape[0]
    dec_blocks = sorted({int(m.group(1)) for k in mapped
                         if (m := re.match(r"dec\.(\d+)\.", k))})
    depth = len(dec_blocks)
    reg = mapped["dec.reg_token"].shape[2]
    meta = {
        "general.architecture": "pi3x",
        "general.file_type": FILE_TYPE[args.outtype],
        "pi3x.patch_size": 14,
        "pi3x.embed_dim": int(emb),
        "pi3x.num_heads": 16,
        "pi3x.enc_depth": 24,
        "pi3x.dec_depth": depth,
        "pi3x.num_register_tokens": int(reg),
        "pi3x.head_depth": 5,
        "pi3x.img_mean": [0.485, 0.456, 0.406],
        "pi3x.img_std": [0.229, 0.224, 0.225],
        "pi3x.notes": "images-only build (multimodal conditioning weights "
                      "dropped); point/conf heads are ConvHead; metric head "
                      "= exp(mth(raw)); poses from SVD orthogonalization on "
                      "the host; conv-head normals/uv per blueprint",
    }

    if args.dry_run:
        for k in sorted(mapped):
            print(k, tuple(mapped[k].shape))
        return

    out = GGUFWriter(args.output, arch="pi3x")
    for k, v in meta.items():
        if isinstance(v, list):
            out.add_array(k, v)
        elif isinstance(v, int):
            if k == "general.file_type":
                out.add_uint32(k, v)
            else:
                out.add_uint32(k, v)
        else:
            out.add_string(k, v)

    def t_to_np(t):
        return t.cpu().contiguous().numpy()

    for i, (k, t) in enumerate(mapped.items()):
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
