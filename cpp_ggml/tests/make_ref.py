#!/usr/bin/env python3
"""Generate the torch reference for test_s2_independence.

Reproduces the exact deterministic LCG frames the C++ test builds
(seed 7, S=2, H=W=256), runs the official f32 model, and writes
<out>.pose.bin / <out>.depth.bin in the CLI binary format.
"""
import sys
from pathlib import Path

import numpy as np
import torch

CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "third_party" / "vggt-omega-src"))
from vggt_omega.models import VGGTOmega  # noqa: E402


def lcg_frames(S, H, W, seed):
    st = seed
    out = np.zeros((S, 3, H, W), dtype=np.float32)
    for s in range(S):
        for c in range(3):
            flat = out[s, c]
            for i in range(H * W):
                st = (st * 1664525 + 1013904223) & 0xFFFFFFFF
                flat[i // W, i % W] = (st >> 8) / float(1 << 24)
    return out


def main() -> None:
    ckpt = CPP / "models" / "pytorch" / "vggt_omega_1b_512.pt"
    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mapggml_test_ref"

    H = W = 256
    frames = lcg_frames(2, H, W, 7)
    frames.tofile(out_prefix + ".frames.bin")

    m = VGGTOmega(autocast=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model", sd.get("state_dict", sd)))
    m.eval()
    with torch.inference_mode():
        preds = m(torch.from_numpy(frames).unsqueeze(0))
    preds["pose_enc"].float().numpy().reshape(2, 9).astype(
        np.float32).tofile(out_prefix + ".pose.bin")
    preds["depth"].float().numpy().astype(np.float32).tofile(
        out_prefix + ".depth.bin")
    print(f"torch ref written: {out_prefix}.pose.bin / .depth.bin "
          f"(frames: {out_prefix}.frames.bin)")


if __name__ == "__main__":
    main()
