#!/usr/bin/env python3
"""Convert the Pi3 checkpoint (yyfz233/Pi3, model.safetensors) to GGUF for
the mapggml C++ runtime.

Key mapping (torch -> mapggml GGUF):
  encoder.*                   -> bb.*      (DINOv2 ViT-L/14-reg backbone; the
                                             bb.* prefix reuses the VGGT
                                             builder's encoder verbatim)
  register_token              -> dec.reg_token
  decoder.N.*                 -> dec.N.*   (36 alternating BlockRope blocks)
  point_decoder.*             -> pdec.*    (TransformerDecoder: project + 5 blocks + out)
  point_head.*                -> phead.*   (LinearPts3d: linear + pixel-shuffle 14)
  conf_decoder.*              -> cdec.*
  conf_head.*                 -> chead.*
  camera_decoder.*            -> mdec.*
  camera_head.*               -> mhead.*   (raw fc_t/fc_rot; SVD orth on the host)
  image_mean / image_std      -> pi3.img_mean / pi3.img_std metadata

Architecture metadata written: general.architecture="pi3", patch_size=14,
embed_dim=1024, num_heads=16, enc_depth=24, dec_depth=36, head_depth=5,
num_register_tokens=5 (decoder registers; the encoder carries its own 4
DINOv2 registers internally), rope freq 100.

Usage:
  python3 scripts/convert_pi3_to_gguf.py models/pytorch/pi3/model.safetensors \
      models/gguf/pi3-f16.gguf --outtype f16
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
    "q6_K": (GGMLQuantizationType.Q6_K, 256, 210),
    "q5_K": (GGMLQuantizationType.Q5_K, 256, 176),
}

FILE_TYPE = {
    "f32": 0, "f16": 1, "q8_0": 8, "q6_K": 15, "q5_K": 13,
}

# 2-D Linear weights get quantized; everything else stays at the outtype
# float (biases, norms, layer scales, token params, pos_embed).
_LINEAR_PAT = re.compile(
    r"\.(attn\.(qkv|q|k|v)|attn\.proj|mlp\.fc1|mlp\.fc2|projects|linear_out|proj|"
    r"res_conv[123]|more_mlps\.[02]|fc_t|fc_rot)\.weight$")


def is_linear_weight(key: str) -> bool:
    return bool(_LINEAR_PAT.search(key))


def rename_key(k: str) -> str:
    if k.startswith("encoder."):
        return "bb." + k[len("encoder."):]
    if k == "register_token":
        return "dec.reg_token"
    if k.startswith("decoder."):
        return "dec." + k[len("decoder."):]
    if k.startswith("point_decoder."):
        return "pdec." + k[len("point_decoder."):]
    if k.startswith("point_head."):
        return "phead." + k[len("point_head."):]
    if k.startswith("conf_decoder."):
        return "cdec." + k[len("conf_decoder."):]
    if k.startswith("conf_head."):
        return "chead." + k[len("conf_head."):]
    if k.startswith("camera_decoder."):
        return "mdec." + k[len("camera_decoder."):]
    if k.startswith("camera_head."):
        return "mhead." + k[len("camera_head."):]
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


def count_prefix(sd: dict, prefix: str, level: int) -> int:
    # count unique block indices at the given dot-level (e.g. encoder.blocks.N)
    return len({k.split(".")[level] for k in sd
                if k.startswith(prefix) and k.count(".") > level})


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
    sd = load_file(args.checkpoint)
    assert isinstance(sd, dict) and sd, "unexpected checkpoint format"

    # Map names; split fused qkv (the C++ attention consumes separate q/k/v).
    mapped, dropped = {}, []
    for k, t in sd.items():
        if k in ("image_mean", "image_std"):
            continue  # carried into the metadata below
        nk = rename_key(k)
        assert nk != k or k.startswith(("enc.", "dec.", "pdec.", "phead.",
                                        "cdec.", "chead.", "mdec.", "mhead.")), \
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
    print(f"tensors: {len(mapped)} mapped, {len(dropped)} dropped")

    mean = sd["image_mean"].flatten().tolist()
    std = sd["image_std"].flatten().tolist()
    meta = {
        "general.name": "pi3",
        "general.architecture": "pi3",
        "pi3.dtype": args.outtype,
        "pi3.patch_size": 14,
        "pi3.embed_dim": 1024,
        "pi3.num_heads": 16,
        "pi3.enc_depth": count_prefix(mapped, "bb.blocks.", 2),
        "pi3.dec_depth": count_prefix(mapped, "dec.", 1),
        "pi3.head_depth": 5,          # TransformerDecoder blocks (fixed)
        "pi3.num_register_tokens": 5,  # decoder register tokens
        "pi3.rope_freq": 100,
        # The model normalizes with the ImageNet constants (buffers).
        "pi3.img_mean": mean,
        "pi3.img_std": std,
        "pi3.notes": "camera head outputs raw fc_t/fc_rot; the host does the "
                     "SVD orthogonalization (pi3.models.layers.camera_head)",
    }

    if args.dry_run:
        for k, v in meta.items():
            print(f"  {k} = {v}")
        print(f"would write {len(mapped)} tensors to {args.output}")
        return

    out = GGUFWriter(args.output, "pi3")
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
