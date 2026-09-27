#!/usr/bin/env python3
"""Convert the VGGT-Omega checkpoint (.pt) to GGUF for mapggml.

Usage:
  python3 convert_vggt_omega_to_gguf.py \
      cpp_ggml/models/pytorch/vggt-omega/vggt_omega_1b_512.pt \
      cpp_ggml/models/gguf/vggt-omega-1b-512-f16.gguf --outtype f16

Supported checkpoints (facebook/VGGT-Omega):
  vggt_omega_1b_512.pt            -> vggt-omega-1b-512
  vggt_omega_1b_416_reproduce.pt  -> vggt-omega-1b-416-reproduce
  vggt_omega_1b_256_text.pt       -> vggt-omega-1b-256-text (keeps the extra
      text_alignment_head.* tensors as f32/quantized extras; the C++ graph
      ignores them, matching the Python default enable_alignment=False path)

Checkpoint-specific handling:
  - LinearKMaskedBias: every `*.attn.qkv.bias_mask` zeroes the K-bias third.
    We pre-multiply the bias and drop the mask, so the C++ graph sees a
    plain Linear.
  - LayerNorm/conv/rope-period buffers stay floating point under quantized
    outtypes; only 2-D Linear weights are quantized (q8_0/q5_K).

GGUF dimension convention: gguf-py reverses numpy shape when writing, so a
torch Linear weight {out, in} lands as ggml ne {in, out} and a Conv2d kernel
{OC, IC, KH, KW} as ne {KW, KH, IC, OC} — exactly what ggml ops expect.
"""

import argparse
import ctypes
import os
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType
from gguf import quants

QK8_0 = 32
QK_K = 256

QTYPES = {
    "q8_0": GGMLQuantizationType.Q8_0,
    "q5_K": GGMLQuantizationType.Q5_K,
}

FILE_TYPE = {
    "f32": 0, "f16": 1, "q8_0": 8, "q5_K": 13,
}

# (block_size, type_size) for the ctypes K-quant path (ggml-quants.h):
# all K-quants block over QK_K = 256 elements
KBLK = {
    GGMLQuantizationType.Q5_K: (256, 176),
}


def unwrap_checkpoint(obj):
    """Accept {model: {...}} / {state_dict: {...}} / bare state_dict."""
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def rename_key(k: str) -> str:
    """Map torch parameter names to the mapggml GGUF names."""
    if k.startswith("aggregator.patch_embed."):
        return "bb." + k[len("aggregator.patch_embed."):]
    if k.startswith("aggregator.frame_blocks."):
        rest = k[len("aggregator.frame_blocks."):]
        idx, tail = rest.split(".", 1)
        return f"agg.frame.{idx}.{tail}"
    if k.startswith("aggregator.inter_frame_blocks."):
        rest = k[len("aggregator.inter_frame_blocks."):]
        idx, tail = rest.split(".", 1)
        return f"agg.inter.{idx}.{tail}"
    if k == "aggregator.rope_embed.periods":
        return "agg.rope_periods"
    if k == "aggregator.camera_token":
        return "agg.cam_token"
    if k == "aggregator.register_token":
        return "agg.reg_token"
    if k.startswith("camera_head."):
        return "cam." + k[len("camera_head."):]
    if k.startswith("dense_head."):
        return "dp." + k[len("dense_head."):]
    return k


def apply_k_bias_mask(sd: dict) -> dict:
    """Fold LinearKMaskedBias masks into the biases; drop the masks."""
    out = {}
    masks = {k for k in sd if k.endswith(".attn.qkv.bias_mask")}
    for k, v in sd.items():
        if k in masks:
            continue
        if k.endswith(".attn.qkv.bias"):
            mask_key = k + "_mask"
            if mask_key in sd:
                v = v * sd[mask_key]
        out[k] = v
    return out


def quantize_k(ctypes_lib, t, qt):
    """Quantize a (out,in) f32 matrix via ggml's ggml_quantize_chunk (ctypes).

    gguf-py's pure-python quantizer lacks the K-quants; libggml-cpu.so built
    in build-cpu/lib provides them. Returns the uint8 byte image of shape
    (out, in//block*type_size), matching gguf-py's raw-shape convention.
    """
    rows, cols = t.shape
    blck, tsize = KBLK[qt]
    dst = np.empty(rows * cols // blck * tsize, dtype=np.uint8)
    src = np.ascontiguousarray(t)
    n = ctypes_lib.ggml_quantize_chunk(
        int(qt),
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.c_void_p),
        0, rows, cols, None)
    assert n == dst.size, (n, dst.size)
    return dst.reshape(rows, cols // blck * tsize)


def find_ggml_lib() -> str:
    """Locate libggml-cpu.so for the ctypes K-quant path.

    Probes $MAPGGML_BUILD_DIR first, then every build-*/lib under the
    cpp_ggml root (CPU builds always contain the K-quant quantizers).
    """
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
    return lib


def model_name_from_ckpt(path: str) -> str:
    """vggt_omega_1b_416_reproduce.pt -> vggt-omega-1b-416-reproduce"""
    stem = Path(path).stem
    if stem.startswith("vggt_omega_"):
        stem = stem[len("vggt_omega_"):]
    return "vggt-omega-" + stem.replace("_", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--outtype", default="f16",
                    choices=list(FILE_TYPE.keys()))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the mapping without writing the GGUF")
    args = ap.parse_args()

    ctypes_lib = None
    if args.outtype in QTYPES:
        ctypes_lib = load_ggml_lib()

    print(f"loading {args.checkpoint} ...")
    sd = unwrap_checkpoint(torch.load(args.checkpoint, map_location="cpu",
                                      weights_only=False))
    sd = {(k[7:] if k.startswith("module.") else k): v
          for k, v in sd.items() if isinstance(v, torch.Tensor)}
    sd = apply_k_bias_mask(sd)
    # split packed qkv Linears into separate q/k/v so the C++ graph needs no
    # strided views over the projection output
    split = {}
    for k, v in sd.items():
        if k.endswith(".attn.qkv.weight") and v.dim() == 2:
            p = k[: -len("qkv.weight")]
            rows_n = v.shape[0] // 3
            for name, rows in (("q", 0), ("k", 1), ("v", 2)):
                split[f"{p}{name}.weight"] = v[rows * rows_n:(rows + 1) * rows_n]
        elif k.endswith(".attn.qkv.bias"):
            p = k[: -len("qkv.bias")]
            rows_n = v.shape[0] // 3
            for name, rows in (("q", 0), ("k", 1), ("v", 2)):
                split[f"{p}{name}.bias"] = v[rows * rows_n:(rows + 1) * rows_n]
        else:
            split[k] = v
    sd = split
    print(f"{len(sd)} tensors after qkv split")

    mapped = {rename_key(k): v for k, v in sd.items()}

    # Derive architecture facts from the tensors themselves (so the 512 /
    # 416-reproduce / 256-text 1B checkpoints all convert without edits).
    patch_size = 16
    num_register_tokens = int(mapped["agg.reg_token"].shape[2])
    embed_dim = int(mapped["agg.reg_token"].shape[3])
    aa_depth = 1 + max(int(k.split(".")[2]) for k in mapped
                       if k.startswith("agg.frame."))
    depth = 1 + max(int(k.split(".")[2]) for k in mapped
                    if k.startswith("bb.blocks."))
    num_heads = embed_dim // int(mapped["agg.frame.0.attn.q_norm.weight"]
                                 .shape[0])
    has_text_head = any(k.startswith("text_alignment_head.") for k in mapped)
    name = model_name_from_ckpt(args.checkpoint)
    # training resolution follows the official checkpoint suffix
    res_digits = "".join(ch for ch in name.split("-")[-2] if ch.isdigit())
    image_resolution = int(res_digits) if res_digits else 512
    print(f"patch={patch_size} embed_dim={embed_dim} depth={depth} "
          f"aa_depth={aa_depth} heads={num_heads} "
          f"num_register_tokens={num_register_tokens} "
          f"image_resolution={image_resolution} text_head={has_text_head}")

    if args.dry_run:
        for k in sorted(mapped):
            v = mapped[k]
            print(f"{k:64s} {tuple(v.shape)} {v.dtype}")
        return

    w = GGUFWriter(args.output, arch="vggt")
    w.add_string("general.name", name)
    w.add_string("general.architecture", "vggt")
    w.add_string("vggt.dtype", args.outtype)
    w.add_uint32("vggt.patch_size", patch_size)
    w.add_uint32("vggt.embed_dim", embed_dim)
    w.add_uint32("vggt.depth", depth)
    w.add_uint32("vggt.aa_depth", aa_depth)
    w.add_uint32("vggt.num_heads", num_heads)
    w.add_uint32("vggt.num_register_tokens", num_register_tokens)
    w.add_uint32("vggt.trunk_depth", 4)
    w.add_uint32("vggt.dpt_features", 256)
    w.add_uint32("vggt.image_resolution", image_resolution)
    w.add_bool("vggt.enable_text_alignment", has_text_head)
    w.add_array("vggt.cached_layer_idx", [4, 11, 17, 23])
    w.add_array("vggt.register_attn_block_idx", [2, 6, 9, 14, 20])
    w.add_array("vggt.img_mean", [0.485, 0.456, 0.406])
    w.add_array("vggt.img_std", [0.229, 0.224, 0.225])

    n_q = 0
    for name in sorted(mapped):
        t = mapped[name].to(torch.float32).contiguous().numpy()
        is_linear_weight = name.endswith(".weight") and t.ndim == 2
        if args.outtype in QTYPES and is_linear_weight:
            if args.outtype == "q8_0":
                block = QK8_0
                if t.shape[1] % block != 0:
                    print(f"warning: {name}: K={t.shape[1]} not divisible by "
                          f"{block}; keeping f32")
                    w.add_tensor(name, t)
                    continue
                qt = QTYPES[args.outtype]
                raw = quants.quantize(t, qt)
                # raw is the uint8 byte image (rows, in//block*type_size);
                # the writer converts it back to the logical shape and reverses
                # to ggml ne on write, so raw_shape must be raw.shape
                w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=qt)
            else:  # K-quants via ggml_quantize_chunk (gguf-py lacks them)
                qt = QTYPES[args.outtype]
                blck, _tsize = KBLK[qt]
                if t.shape[1] % blck != 0:
                    print(f"warning: {name}: K={t.shape[1]} not divisible by "
                          f"{blck}; keeping f32")
                    w.add_tensor(name, t)
                    continue
                raw = quantize_k(ctypes_lib, t, qt)
                w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=qt)
            n_q += 1
        elif args.outtype == "f16" and name.endswith(".weight") and t.ndim >= 2:
            # f16 only for *.weight tensors (2-D Linear matrices and 4-D conv
            # kernels). Everything else — biases, norms, gammas, 4-D token
            # parameters (cam/register/cls/storage), rope periods — stays f32
            # so graph arithmetic never mixes dtypes.
            w.add_tensor(name, t.astype(np.float16))
        else:
            w.add_tensor(name, t)

    print(f"quantized matrices: {n_q}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
