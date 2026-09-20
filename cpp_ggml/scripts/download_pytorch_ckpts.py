#!/usr/bin/env python3
"""Download the official upstream checkpoints into models/pytorch/.

Requires a HuggingFace token only for the gated facebook/VGGT-Omega repo
(~/.cache/huggingface/token or $HF_TOKEN); the other repos are public.

Usage:
  python3 scripts/download_pytorch_ckpts.py [--dir models/pytorch] [--only NAME ...]
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

# (dest-relpath, hf_repo, hf_filename).  dest-relpath is what the converters
# and run_mapggml.sh expect; the mirror-style retry in run_mapggml.sh applies
# to every entry via HF_ENDPOINT.
CKPTS = [
    # facebook/VGGT-Omega (gated): three official checkpoints
    ("vggt_omega_1b_512.pt",            "facebook/VGGT-Omega", "vggt_omega_1b_512.pt"),
    ("vggt_omega_1b_416_reproduce.pt",  "facebook/VGGT-Omega", "vggt_omega_1b_416_reproduce.pt"),
    ("vggt_omega_1b_256_text.pt",       "facebook/VGGT-Omega", "vggt_omega_1b_256_text.pt"),
    # official VGGT-1B (facebookresearch)
    ("vggt1b/model.pt",                 "facebook/VGGT-1B",    "model.pt"),
    # pi3 / pi3x (yyfz233)
    ("pi3/model.safetensors",           "yyfz233/Pi3",         "model.safetensors"),
    ("pi3x/model.safetensors",          "yyfz233/Pi3X",        "model.safetensors"),
    # mapanything (facebook/map-anything; snapshot keeps config.json for
    # MapAnything.from_pretrained, so download both files)
    ("mapanything/model.safetensors",   "facebook/map-anything", "model.safetensors"),
    ("mapanything/config.json",         "facebook/map-anything", "config.json"),
    # dust3r (naver; official 512_dpt release — M5). The HF repo ships
    # model.safetensors + config.json (PyTorchModelHubMixin layout), NOT a
    # .pth; from_pretrained(<dir>) reads both (spy sets landscape_only off
    # via the config patch, matching the official load_model behaviour).
    ("dust3r/model.safetensors",
     "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt", "model.safetensors"),
    ("dust3r/config.json",
     "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt", "config.json"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent / "models" / "pytorch"))
    ap.add_argument("--only", nargs="*", default=None, help="subset of checkpoint filenames")
    args = ap.parse_args()

    wanted = CKPTS
    if args.only:
        wanted = [e for e in CKPTS
                  if any(e[0].startswith(o) or o in e[0] for o in args.only)]
        assert wanted, f"--only matched no entries: {args.only}"
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dest_rel, repo, fname in wanted:
        dest = out_dir / dest_rel
        if dest.exists() and dest.stat().st_size > 1024 ** 2:
            print(f"[skip] {dest} already present")
            continue
        print(f"[down] {repo}/{fname} -> {dest}")
        got = hf_hub_download(repo, fname, local_dir=str(out_dir))
        # hf_hub_download returns <local_dir>/<repo-path>; relocate to the
        # registry layout when the repo file name differs from our dest name
        got_path = Path(got)
        if got_path != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            got_path.replace(dest)
        print(f"[done] {dest}")


if __name__ == "__main__":
    main()
