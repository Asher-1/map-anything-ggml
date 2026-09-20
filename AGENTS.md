# AGENTS.md — cpp_ggml multi-model end-to-end inference: extension guide and iron rules

> Audience: AI agents / developers extending this repo with new models
> (the VGGT paradigm: DINOv2-style encoder + globally-attentive decoder +
> multiple output heads) into ggml C++ end-to-end inference.
> This file records only **field-proven** rules; implementation details live
> in `cpp_ggml/FEATURE_PARITY_AUDIT.md` (alignment status and the
> cross-model metric table), and the per-model `benchmarks/results/*/`
> reports.

---

## 0. Five iron rules (violating any of them means redoing the work)

1. **Every change to third_party/ggml must go through a git patch**
   (`cpp_ggml/third_party/ggml-patches/NNNN-*.patch`, applied idempotently
   at CMake configure time by `scripts/apply_ggml_patches.sh`).
   Hand-editing the vendored ggml sources is forbidden. Verification: inside
   `third_party/ggml`, `git apply --check --reverse ../ggml-patches/*.patch`
   passing fully <=> the worktree's dirty diff matches the patch set
   exactly. The repo's own code (`src/*.cpp`) is committed normally and
   does not go through patches.
2. **No gate, no "done".** An integration counts as done only when the
   full five-piece suite passes: the gate matrix (quantizations x backends)
   + the official-protocol bench + (if the paper defines one) the paper
   protocol eval + the real-scene reconstruction comparison + the speed
   comparison. Numbers must be measured on both sides (official torch vs
   cpp); quoting one side or upstream prose is forbidden.
3. **The comparison protocol must be aligned item-by-item with the baseline
   implementation** (scale alignment, masks, resolution, sampling window).
   This repo's reconstruction comparisons always use avg_dis scale
   alignment + view0-frame point clouds (the pi3 family takes depth from
   `local_points[...,2]`, the vggt family from its dedicated depth head).
   Skipping the alignment makes even the torch reference "fail" (pi3x once
   showed a bogus AbsRel of 0.52 this way).
4. **Layout first**: before writing any numeric comparison, write both
   sides' tensor memory layouts verbatim in the script docstring (pitfall
   #10: ggml's fastest axis is ne[0]; the numpy equivalent of {W,H,C,S} is
   (S,C,H,W); for square grids the flat torch/cpp bytes coincide, which
   hides axis errors).
5. **Chain of numeric evidence**: every conclusion needs dump/replay/
   identity-probe support; a self-contradiction like "corr ~0 but the final
   corr is 0.9999" means the comparison script's layout is wrong, not the
   model.

---

## 1. Unified architecture (registry pattern, same as mapanything)

```
src/graph_builder.hpp   IGraphBuilder interface + the unified ModelOutputs struct
src/vggt_graph.cpp      one Impl per model + static RegisterBuilder at the file bottom
src/gguf_loader.cpp     per-architecture metadata branch keyed on general.architecture
src/cli.cpp             arch-agnostic: read inputs -> builder.build -> compute ->
                        write .pose/.depth/.conf/... files per ModelOutputs family
```

- Each builder = `struct XImpl : IGraphBuilder` plus, at the file bottom,
  `static RegisterBuilder g_reg_x("arch_name", []{...});`.
  New models **add** an Impl and a registration; modifying the public
  behavior of existing Impls is forbidden (backward compatibility).
- Output contracts come in **three** families (where cli.cpp writes files);
  within a family the file names are identical:
  - vggt family: `pose.bin` = pose_enc (S,9), `.depth.bin`,
    `.depth_conf.bin` (+ `.points.bin`/`.points_conf.bin` for vggt-1b's
    point head, + `.text_embedding.bin` for the omega text variant);
  - pi3 family: `pose.bin` = row-major c2w 4x4 (translation at flat
    3/7/11!), `.local_points.bin`, `.conf.bin`, `.depth.bin` (= the
    local_points camera-z; the pi3 family has no dedicated depth head and
    the CLI writes it so downstream viz/npz work uniformly),
    `.points.bin`, `.pose_raw.bin` (pi3x/mapanything also write
    `.scale.bin`; mapanything also `.mask.bin`);
  - dust3r family: no pose (pair-wise; both pointmaps in view1's frame),
    `.local_points.bin`, `.conf.bin`, `.depth.bin`, `.points.bin`.
  A new model must fall into one of these families or explicitly extend
  ModelOutputs + the cli write-out branch.
- Shared components (encoder/rope/decoder/attention_v/block_v) are reused
  through inheritance (Pi3XImpl : Pi3Impl). Changing a shared component
  requires re-running the gates of **all** models.

## 2. New-model onboarding (ten steps, each with acceptance)

| Step | Artifact | Acceptance |
|---|---|---|
| 1. Source repo as submodule | `third_party/<model>-src/` | registered in `.gitmodules` + `check_submodules` probes for key source files (an empty leftover directory does not count); never hand-edit submodule content |
| 2. Checkpoint | `models/pytorch/<model>/` | HF/official download, record the source |
| 3. Converter | `scripts/convert_<model>_to_gguf.py` | dry-run enumerates every key; verify the type distribution with gguf.GGUFReader (see §3) |
| 4. Builder | a new Impl in `src/vggt_graph.cpp` | the whole graph runs (518x518 S=2 smoke) |
| 5. Spy reference | `scripts/spy_torch_<model>.py` | hook the official intermediates (leaf modules + counters, see pitfall #11) |
| 6. Gate | `scripts/e2e_gate_matrix.sh <arch>` | 4 quants x CPU/CUDA/Vulkan all PASS; thresholds calibrated to that model's noise floor and written into the cmp_gate_matrix.py comments |
| 7. Official bench | `scripts/bench_official_eth3d.py --arch <arch>` | two-sided metric-level parity (130 sets, or 10 first) |
| 8. Paper protocol (if any) | `scripts/eval_pi3_paper_eth3d.py --arch` | two-sided delta < 1e-2; pin the paper numbers when possible |
| 9. Launcher | the `model_info` registry in `run_mapggml.sh` + CKPTS in `scripts/download_pytorch_ckpts.py` | `./run_mapggml.sh demo --models <alias> --dry-run` prints the whole chain; infer/gate/bench pass for the new alias |
| 10. Report | `charts/<model>/` + `results/<model>/` | 3 recon figures + 4 statistical charts + an illustrated report (reuse `compare_reconstruction_pi3x.py --arch` / `compare_reconstruction_vggt.py` / `plot_charts_model.py --arch`) |

Shared tooling (do not reinvent):
`bench_latency.py` (CLI steady-state latency),
`bench_torch_pi3x.py --arch {pi3,pi3x,vggt1b,mapanything,dust3r}` (official
torch baseline, incl. --tf32/--device cpu),
`cmp_gate_matrix.py <arch> <prefix> <quant> [ref] [S H W]` (gate
comparison, branching per arch; ref overrides the session cache path),
`eval_cpp_cli.py` (WAI data/metric helper), `make_predictions_npz.py`
(detects the vggt/pi3/dust3r family contract automatically),
`run_mapggml.sh` (one-click launcher; the model registry is the single
source of truth).

## 3. Converter iron rules (already trodden flat by the existing converters)

- Runtime **drift guard** for the `QTYPES` table: query
  `ggml_type_size/ggml_blck_size` from the loaded libggml via ctypes and
  assert (q8_0's tsize was once written as 20 -> undersized buffers ->
  SIGSEGV inside quantize_row_q8_0_ref).
- `_LINEAR_PAT` must list the `.attn.q/.k/.v.weight` tensors **after the
  qkv split** (missing them -> the attention Linears silently stay F32;
  symptom: the f16 and f32 gguf outputs have identical md5).
- `coerce()` must live inside main() (module level cannot see args); f16
  needs an explicit `t.half()` branch; **everything non-Linear stays F32**
  (token/pos_embed/norm weights in F16 break concat types or precision).
- Quantization skip condition: `t.shape[1] % blck_size != 0` (heads with an
  odd row width stay float).
- Fused-qkv splitting and multimodal-weight dropping (images-only) are
  listed explicitly in the key map.

## 4. High-frequency builder pitfalls (ordered by how often they bit)

1. **`ggml_view_4d` has no nb0** (the innermost axis is implicitly
   contiguous; signature nb1/nb2/nb3/offset, offset last). Bitten four
   times (dpt_vggt, pi3 linear_pts3d, pi3x rep_pad2, the metric kn/kr read).
2. **`ggml_set_input()` only marks, it does not upload data** — every input
   tensor must be `ggml_backend_tensor_set()` in the static upload block
   (pi3x missed the decoder rope tables -> the whole decoder had no
   positions -> rot stuck at 24.85°).
3. **torch GroupNorm defaults to affine=True** — after a GN you must
   mul/add the per-channel weight/bias; broadcasting only works with
   `ne[0]==1`: reshape (C,) into `{1,1,C,1}`; hidden-dim GN is 2C.
4. **Per-head rope tables must be filled looping over the head axis**
   ([row*nh+h]*dh layout; writing only head0 leaves 7/8 of the heads
   position-less). Identity probe: the token at position (0,0) must have
   rope(k)==k — check the identity rows first.
5. **Diff every hand-written static table against the tensor shape formula
   one-to-one** (uv_table once had the channel fastest while the tensor
   expected x fastest, scrambling coordinates with channels).
6. **Row-major c2w translation sits at flat 3/7/11** (last column), not
   12/13/14.
7. **Incremental cmake may reuse stale objects**: verify with
   `strings bin/vggt-cli | grep <new-string>`; `grep -c && cmd` short-circuits
   and skips cmd.
8. **gallocr does not allocate non-main-chain nodes**: a dumped node must be
   set_output AND on the main chain expanded by build_forward_expand; an
   isolated view/cont chain never gets allocated -> all-zero mirages. On
   the torch side, hooks on shared modules (e.g. one RoPE2D serving every
   block) need an incrementing counter, otherwise last-write-wins.
9. **Multiple build directories go stale**: after adding a
   builder/registration, build-cpu/cuda/vulkan/dump must **all** be rebuilt
   (debugging only in build-dump = the other backends throw "unknown
   architecture" at runtime).
10. **Layout reinterpretation**: numpy's C-order last axis is fastest <=>
    ggml's ne[0] is fastest; reinterpreting a dump requires reversing the
    axis order (np.fromfile().reshape((d,c,b,a)) corresponds to ne{a,b,c,d}).

## 5. Debugging methodology (proven localization chain)

```
a known-good model (same builder code) as the calibration control ->
f32/f16 gguf md5 discrimination (identical = the converter silently
degraded) -> official-module replay (feed the captured inputs through the
official implementation) -> identity probes (rope (0,0) rows kr==kn etc.)
-> per-operator dumps (layout first, §0-4)
```

GPU resources: when running torch and then the cpp CLI inside one process,
you must `del model; gc.collect(); torch.cuda.empty_cache()`, otherwise the
cpp side's cudaMalloc OOM gets misread as a porting bug. Never run
latency/benchmark measurements concurrently with builds or other GPU/CPU
loads — a concurrent full-core build once contaminated the CPU rows by 42%.

## 6. Gating and measurement principles

- A gate threshold = that model's own noise floor x margin, not a
  cross-model constant (pi3x's floor is 0.015: flash_attn_ext's K/V are
  always f16, so the f16 and f32 weight floors coincide; record q5_K's
  degradation factor honestly and recommend a quant).
- **The ggml CPU default thread count = GGML_DEFAULT_N_THREADS == 4**! The
  CLI now defaults to `hardware_concurrency` (override with `--threads`;
  thread-count independence verified bit-exactly). Confirm the thread count
  before any CPU comparison (pi3x once yielded the wrong conclusion "torch
  CPU is faster"; after the correction cpp is ahead 1.15-1.35x).
- Latency measurement: CLI `--timing --warmup N --repeats M` (in-process
  steady state + CUDA graph replay); wrap the torch side in
  `torch.cuda.synchronize()`; declare fp32-faithful or TF32 (`--tf32`).
  The timing JSON is a compact format that changes fields with the repeats
  mode — grab one raw output before writing a parser.
- Speed comparisons must have exclusive GPU/CPU (staggered queues + done
  files).

## 7. Chart and document pipeline (one set per model, into charts/<model>/)

```
latency json(s) + torch baseline json ─┐
gate matrix log ───────────────────────┼─> scripts/plot_charts_model.py --arch <a>
torch/cpp f16 output dumps ────────────┘     -> e2e_latency_bar / pose_error_heatmap
                                             / quant_pareto_3d / parity_scatter
                                             / gate_summary.json
real-scene window (courtyard [5,0]) ───> scripts/compare_reconstruction_{pi3x,vggt}.py
                                             -> recon_{depth,pointcloud,metrics}_comparison.png
                                             + recon_comparison.md
all of the above ──────────────────────> charts/<model>/<model>_report.md (illustrated)
```

## 8. Integrated models and current status (2026-09-24)

| Model | gate | official bench | paper protocol | recon/charts | speed vs official |
|---|---|---|---|---|---|
| vggt-omega | all PASS (4 quants x 3 backends) + 256-text (cos 1.0) | ✅ 130 sets (AUC5 79.2 vs official 79.53) | — (no paper protocol) | ✅ | CUDA 0.87x (first port, strong official baseline; see omega_report.md) |
| vggt-1b | 12/12 (+points output, S=4) | ✅ 130 sets | — | ✅ | CUDA 1.60-1.69x |
| pi3 | 12/12 | ✅ 130 sets | ✅ 13 scenes pinned | ✅ | CUDA on par (q8_0 1.05x) |
| pi3x | 12/12 | ✅ 130 sets | ✅ 13 scenes <= 0.0021 | ✅ | CUDA 1.44-1.50x / Vulkan 1.21x / CPU 1.15-1.35x |
| mapanything | 9/9 | ✅ 130 sets | — (the official protocol IS its paper protocol) | ✅ (AbsRel delta < 0.13%) | CUDA 1.64-2.23x / Vulkan 1.67x / CPU q5_K 1.10x |
| dust3r | 12/12 | ✅ 130 sets (pose via closed-form Procrustes) | — (the official protocol is its protocol) | ✅ | CUDA 1.71-1.75x / CPU f16 1.16x |

Model selection: vggt-omega is the default (best accuracy); mapanything
for metric scale/poses; dust3r for resource-constrained devices or
throughput. See the guide at the top of `run_mapggml.sh`.
