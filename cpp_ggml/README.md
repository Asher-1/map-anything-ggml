# cpp_ggml — six 3D-vision models in pure C++ (ggml 0.24.0)

Pure C++ ggml inference for six feed-forward 3D-vision models, with zero
PyTorch dependency at runtime:

| Model | Alias | Family / paradigm | Outputs |
|---|---|---|---|
| [facebook/VGGT-Omega](https://github.com/facebookresearch/vggt-omega) | `512` `416` `text` | vggt (N-view) | camera pose_enc, depth, depth_conf, text-alignment embedding |
| [facebook/VGGT-1B](https://github.com/facebookresearch/vggt) | `vggt-1b` | vggt (N-view) | camera, depth, depth_conf, **world pointmap + conf** |
| [yyfz/Pi3](https://github.com/yyfz/Pi3) | `pi3` | pi3 (N-view) | c2w poses, local pointmaps, conf, depth |
| Pi3X | `pi3x` | pi3 + metric scale | + `.scale.bin` |
| [facebook/map-anything](https://github.com/facebookresearch/map-anything) | `mapanything` | unified (N-view, AAT) | + rays, non-ambiguous mask, metric scale |
| [naver/DUSt3R](https://github.com/naver/dust3r) | `dust3r` | pair-wise ancestor | two pointmaps in view1's frame, conf, depth (no pose head) |

ggml is pinned as a git submodule at v0.24.0 (`third_party/ggml` @
`456172e`); each model's official source is attached under
`third_party/*-src` (reference only, for parity checks). All source-tree
modifications to ggml travel as git patches
(`third_party/ggml-patches/*.patch`, applied automatically by CMake at
configure time) so a fresh clone reproduces the build exactly.

## Model selection guide

Accuracy below is the official MapAnything ETH3D protocol (130 sets, seed
777, 2 views, two-sided vs the official PyTorch — full table in
[FEATURE_PARITY_AUDIT.md](FEATURE_PARITY_AUDIT.md)):

| Model | pointmaps↓ | depth↓ | ATE↓ | rot°↓ | metric scale↓ | speed vs official torch |
|---|---|---|---|---|---|---|
| **vggt-omega** (default) | **0.0302** | **0.0208** | **0.0065** | **0.56** | 0.762 | CUDA 0.87x |
| pi3x | 0.0533 | 0.0369 | 0.0162 | 2.11 | 0.251 | CUDA 1.44-1.50x |
| mapanything | 0.0553 | 0.0469 | 0.0124 | 0.92 | **0.162** | CUDA 1.64-2.23x |
| pi3 | 0.0650 | 0.0485 | 0.0172 | 1.82 | 0.733 | CUDA ~1.0x |
| vggt-1b | 0.0674 | 0.0541 | 0.0208 | 2.28 | 0.788 | CUDA 1.60-1.69x |
| dust3r | 0.1008 | 0.1033 | 0.0509 | 2.77 | 0.897 | **CUDA 1.71-1.75x** |

- **vggt-omega (default)** — best accuracy across the board; the
  256-text variant also emits language-aligned embeddings (`.text_embedding.bin`).
- **mapanything** — pick for metric scale and camera poses (metric_scale
  0.162 far ahead; also the fastest large model on CUDA).
- **dust3r** — pick for resource-constrained devices or maximum throughput
  (pair-wise S=2 only; the family's mildest quantization degradation).
- **pi3x** — balanced alternative with a pinned paper protocol.

Every model passes the quantization x backend gate matrix (f32/f16/q8_0/q5_K
x CPU/CUDA/Vulkan; mapanything ships three quants). Charts and per-model
numbers: [benchmarks/README.md](benchmarks/README.md).

## Quick start

```bash
./run_mapggml.sh demo --models 512            # download -> convert -> build -> infer
./run_mapggml.sh demo --models dust3r --dry-run   # print the whole chain first
./run_mapggml.sh gate --models mapanything --quants f16
./run_mapggml.sh bench --models pi3x --quants all
./run_mapggml.sh help                          # subcommands + model selection guide
```

`demo` performs the whole chain automatically (submodules are initialized on
demand; missing python deps are named explicitly). Manual steps:

```bash
cd cpp_ggml
python3 scripts/download_pytorch_ckpts.py --only <model>   # official ckpt(s)
python3 scripts/convert_<model>_to_gguf.py <ckpt> models/gguf/<stem>.gguf --outtype f16
cmake -B build-cpu && cmake --build build-cpu -j           # or use the launcher
```

Each model directory under `models/MODEL_CARDS.md` documents its ckpt,
default resolution, use cases and the official-protocol artifacts.

## Inference

```bash
# Raw frames (S*3*H*W float32, torch (S,3,H,W) layout, [0,1]) — parity path
./build-cpu/bin/vggt-cli --model models/gguf/vggt-omega-1b-512-f16.gguf \
  --bin frames.bin --H 512 --W 512 --S 2 --out-prefix /tmp/out

# Image input (official "balanced" preprocessing; --image-size per model;
# --resize-mode balanced|max_size)
./build-cpu/bin/vggt-cli --model models/gguf/vggt-omega-1b-416-reproduce-f16.gguf \
  --images a.jpg --images b.jpg --image-size 416 --out-prefix /tmp/out

# --timing: JSON phase breakdown on stderr, mirrored to <prefix>.meta.json
```

Output files are dispatched by the GGUF architecture into **three contract
families** (file names identical within a family; see
`src/cli.cpp` header):

- vggt family: `.pose.bin` (Sx9 trans|quat|fov), `.depth.bin`,
  `.depth_conf.bin` (+ vggt-1b: `.points.bin`/`.points_conf.bin`; + omega
  text: `.text_embedding.bin`);
- pi3 family: `.pose.bin` (row-major c2w 4x4 stack, translation at flat
  3/7/11), `.pose_raw.bin`, `.local_points.bin`, `.conf.bin`, `.depth.bin`
  (= local_points camera-z), `.points.bin` (+ mapanything:
  `.mask.bin`, `.scale.bin`);
- dust3r family: `.local_points.bin` (both views in view1's frame),
  `.conf.bin`, `.depth.bin`, `.points.bin`.

Package into the official demo's `predictions.npz` contract with
`scripts/make_predictions_npz.py --prefix /tmp/out`.

## Parity methodology

Debug dump switch: `MAPGGML_DUMP_STAGE=/dir vggt-cli ...` dumps selected
intermediate nodes as raw f32 bins; pair with the per-model
`scripts/{spy,dump}_torch_*.py` for same-location torch references.

**Memory-order rules for parity** (every false positive so far came from
these):

- a ggml ne `{a,b,c,d}` dump is numpy `reshape(d,c,b,a)` (ne[0] fastest);
- torch `(B, nh, T, dh)` and ggml `{dh, nh, T, B}` dumps are memory-identical;
- `ggml_flash_attn_ext` returns ne `{dh, nh, T, B}`; reshaping to
  `{dim, T, B}` gives head-major token vectors — do **not** permute again.

One-click gates:

```bash
scripts/e2e_gate.sh --quant f16            # omega self-contained (auto torch ref)
scripts/e2e_gate_matrix.sh <arch>          # pi3/pi3x/mapanything/vggt/dust3r
scripts/cmp_gate_matrix.py <arch> <prefix> f16 [ref] [S H W]
```

Gate thresholds are calibrated per model to its own noise floor (see the
comments in `cmp_gate_matrix.py`); all thresholds carry >= 1.7x headroom at
f16.

## Quantization matrix (vggt-omega 512x512x2, CUDA parity reference)

| Format | File size | pose max_abs | depth median_rel |
|-------|---------|--------------|------------------|
| f32   | 4.36 GB | reference | reference |
| f16   | 2.18 GB | 0.0010 | 0.11% |
| q8_0  | 1.25 GB | 0.0022 | 0.33% |
| q5_K  | 0.86 GB | 0.0177 | 0.37% |

q4_K was removed (2026-09-18): its 4.5-bit SNR doubles the pose error vs
q5_K while saving only 13% size — not a valid Pareto point. Per-model
quantization degradation tables live in
[benchmarks/results/gate_matrix.md](benchmarks/results/gate_matrix.md).

## Known numeric differences (not bugs)

Per-op parity all passes (conv/ln/lin/rope/FA/mlp single-op max diff
3e-3~4e-5 with f32 weights). Remaining final-output differences come from
**floating-point summation order** (ggml blocked mul_mat, FA online-softmax
chunking vs torch's order): single-layer FA diff ~4e-5, amplified through 24
transformer layers to ~0.8% mean on features, while the pose head (global
pooling) barely notices (pose 0.0007). f32 and f16 weights produce nearly
identical final outputs, proving this is not a quantization issue.

## Porting notes (hard-won)

1. `ggml_flash_attn_ext` output ne is `{dh,nh,T,B}`, see above;
2. CPU mul_mat with f16 weights x f32 activations misbehaves for some shapes
   -> lin() casts to f32 and flattens to 2-D;
3. Linear bias must be an explicit `ggml_add` (ggml has no fused bias);
4. RoPE row layout `[Ah16|Aw16|Ah16|Aw16]` (cos first half = second half);
   special-token rows are identity (cos=1/sin=0) guaranteed by host tables;
5. the vggt backbone is per-frame attention (S as batch); only inter_frame
   is cross-frame;
6. `ggml_reshape_2d/4d` on contiguous tensors are pure aliases; ne products
   must match memory (64x16x1029 = 1053696 — do not eyeball it);
7. `ggml_gallocr` never reuses/frees memory of nodes marked with
   `ggml_set_output`; the intermediate-dump mechanism relies on this;
8. **strided views cannot be read via `ggml_backend_tensor_get` directly**
   (CPU get is a linear memcpy) — materialize with `ggml_cont` first;
9. CUDA unary/concat kernels require contiguous inputs; CUDA cpy does not
   support K-quant->f32 casts, so K-quant weights go through mul_mat's
   native quantized vec_dot path;
10. dust3r-style DPT must crop refinenet4's output to tap2's grid (odd
    patch grids, e.g. 336/16=21, otherwise hit `ggml_add` can_repeat
    asserts).

## Documentation map

- [FEATURE_PARITY_AUDIT.md](FEATURE_PARITY_AUDIT.md) — feature-parity audit
  vs the official repos + the cross-model metric comparison + the open-items
  list (vggt-1b TrackHead is the next milestone candidate);
- [benchmarks/README.md](benchmarks/README.md) — per-model charts, tables
  and the horizontal comparison;
- [models/MODEL_CARDS.md](models/MODEL_CARDS.md) — per-model ckpt cards;
- [../AGENTS.md](../AGENTS.md) — extension guide for adding new models
  (ten-step checklist, iron rules, pitfalls);
- [benchmarks/RESULTS.md](benchmarks/RESULTS.md) — historical optimization
  records.

## C API for host-application integration (ACloudViewer AICore)

`include/mapggml/capi.h` is the integration surface: a pure C ABI that
mirrors the AICore plugin contract (see ACloudViewer's
`.agents/skills/acloudviewer-aicore-plugin/SKILL.md`) — opaque contexts,
explicit options handles, typed results with release functions,
thread-local `last_error`, borrowed stride-aware image views, and no
environment variables. One context serves all six architectures (the GGUF
`general.architecture` key dispatches internally).

```c
#include "mapggml/capi.h"

mapggml_options* o = mapggml_options_new();
mapggml_options_set_device(o, "auto");          // cpu|cuda|vulkan|metal|auto
void* ctx = mapggml_load("models/gguf/mapanything-f16.gguf", o);
mapggml_options_free(o);

mapggml_result r;
mapggml_run_images(ctx, views, n_views, &r);    // borrowed RGB8/RGBA8/... views
// or the bit-exact parity path:
// mapggml_run_frames(ctx, frames_f32, n, h, w, &r);
// r.pose / r.depth / r.local_points / r.points / r.mask / r.scale —
// family-specific fields are documented in capi.h; free with
mapggml_result_free(&r);
mapggml_ctx_free(ctx);
```

Integration notes:

- `mapggml_image_view` is layout-compatible with AICore's
  `aicore_image_view` (field-by-field conversion in the host adapter).
- `mapggml_patch_size` / `mapggml_default_image_size` let the host pick a
  patch-aligned input size; the checkpoint's own mean/std normalization is
  the graph's first op, so callers feed [0,1] pixels.
- Result fields are architecture-dependent (dust3r has no pose; the text
  embedding only exists on the omega text variant) — branch on
  `r.architecture`; every field's semantics is documented in the header.
- Contexts are not thread-safe: serialize `run_*` calls per context.
- A resolution change rebuilds the runtime (weights re-stage); keeping the
  input geometry across calls reuses everything.

Contract test: `build-cpu/bin/test_capi <model.gguf>` (options NULL-safety,
device enumeration, load-error path, frames-vs-image-view bit equality at
the nominal resolution, and multi-run bit-identical stability).
