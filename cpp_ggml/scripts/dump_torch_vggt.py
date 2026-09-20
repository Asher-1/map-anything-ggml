#!/usr/bin/env python3
"""Dump the official VGGT-1B predictions for frames.bin (torch (S,3,H,W)
f32 in [0,1]) — the parity reference for the vggt builder.

Usage (PYTHONPATH=/tmp/torch_cuda_lib:/tmp/vggt-src:<repo>):
  python3 scripts/dump_torch_vggt.py <frames.bin> <H> <W> <S> <out_prefix>
Writes <out_prefix>.pose.bin / .depth.bin / .depth_conf.bin /
.points.bin / .points_conf.bin (+ .torch_meta.json), all f32.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # CPU-torch
# fallbacks (plain python3 envs) must still produce a valid reference


CPP = Path(__file__).resolve().parent.parent
# official VGGT sources: prefer the repo-local clone (run_mapggml.sh gate
# shallow-clones it into third_party/vggt-src on demand); fall back to a
# checkout pointed at by $VGGT_SRC, then the legacy /tmp/vggt-src location
_vggt_src = next((p for p in (CPP / "third_party" / "vggt-src",
                              Path(__import__("os").environ.get("VGGT_SRC", "")),
                              Path("/tmp/vggt-src")) if p and p.is_dir()), None)
if _vggt_src is None:
    raise SystemExit("official vggt sources not found; run: "
                     "git clone --depth 1 https://github.com/facebookresearch/vggt "
                     "cpp_ggml/third_party/vggt-src  (or set VGGT_SRC)")
sys.path.insert(0, str(_vggt_src))
from vggt.models.vggt import VGGT  # noqa: E402

frames_bin, H, W, S, prefix = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                               int(sys.argv[4]), sys.argv[5])

frames = np.fromfile(frames_bin, dtype=np.float32).reshape(S, 3, H, W)
model = VGGT().eval()
sd = torch.load(CPP / "models/pytorch/vggt1b/model.pt", map_location="cpu",
                weights_only=False)
sd = sd.get("model", sd.get("state_dict", sd))
model.load_state_dict(sd)
model = model.to(DEV)

images = torch.from_numpy(frames).unsqueeze(0).to(DEV)
with torch.inference_mode():
    preds = model(images)

out = {}
for key in ("pose_enc", "depth", "depth_conf", "world_points",
            "world_points_conf"):
    t = preds[key].detach().float().cpu().numpy()
    if t.shape[0] == 1:
        t = t[0]
    out[key] = t
    print(key, t.shape)

pose = out["pose_enc"].reshape(S, 9)
pose.tofile(f"{prefix}.pose.bin")
out["depth"][..., 0].tofile(f"{prefix}.depth.bin")
out["depth_conf"].tofile(f"{prefix}.depth_conf.bin")
out["world_points"].tofile(f"{prefix}.points.bin")
out["world_points_conf"].tofile(f"{prefix}.points_conf.bin")
Path(f"{prefix}.torch_meta.json").write_text(json.dumps(
    {"S": S, "H": H, "W": W, "keys": list(out)}))
print(f"wrote torch reference to {prefix}.*")
