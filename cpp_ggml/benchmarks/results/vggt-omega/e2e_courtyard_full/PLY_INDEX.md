# Full-scene e2e reconstruction — ALL 38 frames, two runtimes (v3, 2026-10-06)

FINAL deliverable for HUMAN inspection. The v3 clouds below are the
**official single-pass semantics**: all 38 frames through ONE forward per
runtime — the model's global attention IS the official multi-frame
consistency mechanism, so there is no chunking, no chunk alignment, and no
chunk-boundary seams at all.

- **Scene/inputs**: ETH3D `courtyard`, ALL 38 frames, official
  benchmarking pack `data/map-anything-benchmarking/eth3d/`.
- **Official answer to "is there a fusion scheme?"**: NO. Official demos
  (demo_gradio.py, visual_util.predictions_to_glb) run one forward over
  all frames and then just concatenate the per-frame filtered clouds —
  the global attention handles consistency internally. Any chunked
  pipeline is an off-official workaround for CPU/small-VRAM boxes.
- **Metric mapping**: one robust Horn Sim3 (all 38 model camera centers vs
  GT centers from scene_meta.json) per runtime into the metric GT world
  frame; residual = the model's own absolute pose error.
- **Point quality = the official visual_util stack**, applied identically
  on both sides (using the torch reference conf so the two clouds stay in
  one-to-one pixel correspondence): isfinite + `depth_edge` (rtol 0.03,
  kills flying pixels at depth discontinuities) + confidence percentile
  **p20 (the official gradio `conf_thres=20` default, floor =
  max(2.0, p) → landed at conf >= 14.1)** + conf-PRIORITY voxel merge (per
  2 cm voxel keep the highest-confidence observation — sharpens surfaces
  that the official plain-concatenate leaves as a blurry multi-view noise
  band; pass `--voxel 0` for the pure official concatenate, whose
  `max_points=300000` linspace truncation is a web-display artifact we
  deliberately do NOT adopt for reconstruction delivery).
- **Reproduce** (defaults now ARE the best pipeline — verified
  byte-identical output on 2026-10-06): `PYTHONPATH=/tmp/torch_cuda_lib:.
  python3 scripts/e2e_full_scene.py --scene-dir <scene> --out-dir <dir>
  --gguf models/gguf/vggt-omega-1b-512-f16.gguf --cli
  build-cuda/bin/vggt-cli` (~1 min on a 4090; on CPU/small-VRAM boxes
  add `--chunk 8 --device cpu`).
- **VRAM fact (measured)**: the 38-frame single forward peaks at only
  **8.75 GiB** on the 24 GB card (4.26 GiB weights + ~118 MiB/frame
  activations, SDPA memory is O(N)) — roughly 100+ frames fit. The
  earlier "38 frames cannot fit in one forward" was an UNVERIFIED
  assumption made while a 17 GB co-tenant held the card; chunking was an
  environment workaround, never a model limit.

## Point clouds (binary PLY, colored, ~4.8 M pts each)

| File | Runtime | Weights |
|---|---|---|
| `torch_f32_global_38f_f16single.ply` | official PyTorch (demo path) | f32 ckpt |
| `cpp_f16single_global_38f.ply` | C++ ggml CLI | **f16** GGUF |

## Verdicts

- Cross-runtime per-point deviation: median **0.015 m** / p95 0.061 m
  over 7.7 M corresponding points (f16 GGUF vs torch f32 — the 6.4 mm
  q6_K-class gate noise amplified by independent metric anchoring).
- GT anchor (single Sim3, 38 camera centers): torch 0.043 m / cpp 0.045 m
  median residual = the model's own absolute pose accuracy on this scene.
- Visual QA (`*.preview.png`): walls close into one coherent building
  with NO chunk seams (single pass), window rows sharp (official filters
  + conf-priority voxel), no ghost walls.
- History: v1 = chunked + chain Sim3 (walls split — chain accumulated the
  6.45→14.12 cross-chunk scale drift); v2 = chunked + per-chunk GT anchor
  (no accumulation, but chunk-boundary seams of ~2x4.5 cm remained and
  the merge was blurrier); v3 = this single-pass version.
