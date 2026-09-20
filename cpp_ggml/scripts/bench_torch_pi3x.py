#!/usr/bin/env python3
"""Latency baseline for the OFFICIAL torch forward of pi3x / pi3 / vggt-1b
(upstream reference for the cpp_ggml speed comparison).

Measures steady-state wall-clock of model(imgs) with warmup, mirroring
bench_latency.py's --warmup/--repeats semantics so the two harnesses are
comparable.  The official model runs fp32 (its native training precision);
note this in the report — cpp f16/q8_0 runs at lower weight precision.

Usage:
  PYTHONPATH=/tmp/torch_cuda_lib:. python3 scripts/bench_torch_pi3x.py \
      --device cuda --repeats 10 --warmup 3 \
      --frames /tmp/frames518.bin --H 518 --W 518 --S 2 \
      --out benchmarks/results/bench_torch_pi3x_cuda.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

CPP = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 matmuls (faster, slightly less precise "
                         "than the official fp32-faithful path)")
    ap.add_argument("--frames", default="/tmp/frames518.bin")
    ap.add_argument("--H", type=int, default=518)
    ap.add_argument("--W", type=int, default=518)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--arch", default="pi3x",
                    choices=["pi3x", "pi3", "vggt1b", "mapanything", "dust3r"],
                    help="official torch harness to run; dust3r is the "
                         "pair-wise naver model (S must be 2, use a /16 "
                         "resolution such as 512; inputs are the raw [0,1] "
                         "frames, the official ImgNorm is applied inside)")
    ap.add_argument("--ckpt", default="",
                    help="checkpoint dir/pt; default per --arch")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    import sys
    if args.arch in ("pi3x", "pi3"):
        sys.path.insert(0, str(CPP / "third_party" / "pi3-src"))
        if args.arch == "pi3x":
            from pi3.models.pi3x import Pi3X as Model
            ckpt = args.ckpt or str(CPP / "models/pytorch/pi3x")
        else:
            from pi3.models.pi3 import Pi3 as Model
            ckpt = args.ckpt or str(CPP / "models/pytorch/pi3")
        frames = np.fromfile(args.frames, dtype=np.float32)
        per = 3 * args.H * args.W
        assert frames.size >= args.S * per, "frames file too small"
        imgs = torch.from_numpy(frames[:args.S * per].copy()) \
                    .reshape(args.S, 3, args.H, args.W).unsqueeze(0)
        model = Model.from_pretrained(ckpt).eval().requires_grad_(False)

        def forward(m, x):
            return m(x)
    elif args.arch == "mapanything":
        # official facebook/map-anything: from_pretrained needs the snapshot
        # dir (config.json); the wrapper normalizes with the encoder's own
        # ImageNet constants and declares data_norm_type=dinov2 (the encoder
        # only ASSERTS the declared normalization)
        import sys
        sys.path.insert(0, str(CPP.parent))
        from mapanything.models.mapanything.model import MapAnything
        ckpt = args.ckpt or str(CPP / "models/pytorch/mapanything")
        frames = np.fromfile(args.frames, dtype=np.float32)
        per = 3 * args.H * args.W
        assert frames.size >= args.S * per, "frames file too small"
        raw = torch.from_numpy(frames[:args.S * per].copy()) \
            .reshape(args.S, 3, args.H, args.W)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        norm = (raw - mean) / std
        model = MapAnything.from_pretrained(ckpt).eval().requires_grad_(False)
        imgs = norm.unsqueeze(0)   # (1, S, 3, H, W): batch of S views

        def forward(m, x):
            views = [{"img": x[:, v], "data_norm_type": ["dinov2"]}
                     for v in range(x.shape[1])]
            return m(views)
    elif args.arch == "dust3r":
        # official naver/dust3r (HF PyTorchModelHubMixin layout: config.json
        # + model.safetensors); pair-wise S=2 forward takes two view dicts,
        # [0,1] imgs normalized to [-1,1] (official ImgNorm (x-0.5)/0.5)
        import sys
        sys.path.insert(0, str(CPP / "third_party" / "dust3r-src"))
        import dust3r.utils.path_to_croco  # noqa: F401
        from dust3r.model import AsymmetricCroCo3DStereo
        assert args.S == 2, "dust3r is pair-wise: S must be 2"
        ckpt = args.ckpt or str(CPP / "models/pytorch/dust3r")
        frames = np.fromfile(args.frames, dtype=np.float32)
        per = 3 * args.H * args.W
        assert frames.size >= args.S * per, "frames file too small"
        raw = torch.from_numpy(frames[:args.S * per].copy()) \
            .reshape(args.S, 3, args.H, args.W) * 2.0 - 1.0
        model = AsymmetricCroCo3DStereo.from_pretrained(ckpt) \
            .eval().requires_grad_(False)
        imgs = raw.unsqueeze(0)   # (1, S, 3, H, W)

        def forward(m, x):
            shape = torch.tensor([[args.H, args.W]], device=x.device)
            views = (dict(img=x[:, 0], true_shape=shape, instance="v1"),
                     dict(img=x[:, 1], true_shape=shape, instance="v2"))
            return m(*views)
    else:  # vggt1b: official facebook/VGGT-1B (repo-local clone preferred;
        # the same candidate order dump_torch_vggt.py uses)
        import sys
        _src = next((p for p in (CPP / "third_party" / "vggt-src",
                                 Path(__import__("os").environ.get("VGGT_SRC", "")),
                                 Path("/tmp/vggt-src")) if str(p) and p.is_dir()), None)
        if _src is None:
            raise SystemExit("official vggt sources not found; git clone "
                             "facebookresearch/vggt into cpp_ggml/third_party/vggt-src")
        sys.path.insert(0, str(_src))
        from vggt.models.vggt import VGGT
        pt = args.ckpt or str(CPP / "models/pytorch/vggt1b/model.pt")
        model = VGGT().eval().requires_grad_(False)
        sd = torch.load(pt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("model", sd.get("state_dict", sd)))
        frames = np.fromfile(args.frames, dtype=np.float32)
        per = 3 * args.H * args.W
        assert frames.size >= args.S * per, "frames file too small"
        imgs = torch.from_numpy(frames[:args.S * per].copy()) \
                    .reshape(args.S, 3, args.H, args.W).unsqueeze(0)

        def forward(m, x):
            return m(x)
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        imgs = imgs.cuda()
        # default: keep matmuls in fp32-faithful mode (official behaviour);
        # --tf32 opts into the faster Ampere+ path for a second baseline
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    else:
        args.device = "cpu"
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 8) // 2))

    with torch.no_grad():
        for _ in range(args.warmup):
            forward(model, imgs)
        if args.device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            forward(model, imgs)
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

    n_par = sum(p.numel() for p in model.parameters())
    report = {
        "harness": f"torch-official-{args.arch}",
        "device": args.device,
        "weight_dtype": "fp32",
        "H": args.H, "W": args.W, "S": args.S,
        "warmup": args.warmup, "repeats": args.repeats,
        "params_M": round(n_par / 1e6, 1),
        "infer_ms_p50": round(statistics.median(times), 1),
        "infer_ms_p95": round(np.quantile(times, 0.95), 1),
        "infer_ms_mean": round(statistics.fmean(times), 1),
        "times_ms": [round(t, 1) for t in times],
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
