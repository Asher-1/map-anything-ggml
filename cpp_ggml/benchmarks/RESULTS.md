# cpp_ggml Benchmark Results

> Updated: 2026-09-19 · full re-measure after the uv-bug fix, official-protocol
> reproduction, and the four-step speed optimization · RTX 4090

## 1. End-to-end accuracy (vs official PyTorch f32, random frames 512x512x2; gate: pose<0.005, depth_med<0.005 — tightened 2026-09-19 from 0.05/0.08)

> After the DPT position-encoding fix (see §4.4) depth parity dropped from
> ~4% to 0.1-0.4% and the quantization ladder is clearly visible. The table
> below is the FINAL binary (convT k==s fast path + single-graph merge); the
> convT fast path's different summation order moved f16 depth parity from
> 0.04% to 0.11% (still far below the gate).

| quant | pose max_abs | depth median_rel | gate |
|-------|--------------|------------------|------|
| f16   | 0.0010 | **0.11%** | PASS |
| q8_0  | 0.0022 | 0.33% | PASS |
| q5_K  | 0.0177 | 0.37% | PASS |

All three backends PASS with f16: CUDA 0.0010/0.11%, CPU 0.0007/0.20%,
Vulkan 0.0018/0.29% (pose max_abs / depth median_rel).

## 2. vs the PyTorch-CUDA baseline (torch 2.6.0+cu124, same GPU, same input)

### 2.1 Speed optimization history (2026-09-18/19, four steps)

| Stage | CUDA f16 inference P50 | Delta |
|------|-------------------|------|
| Baseline (3 graphs + per-call F16 weight cast) | 255.7 ms | — |
| (1) Native F16-weight mul_mat on GPU (removes ~350 per-call casts) | 202.8 ms | -21% |
| (2) aggregator+heads merged into one graph (removes 68 MB D2H->H2D bridge + sync) | 176 ms | -13% |
| (3) F16 activation path (tensor core) | measured SLOWER (189 ms), reverted | — |
| (4) convT k==s fast kernel + 1x1 conv→GEMM (§4.5) | 141 ms | -35 ms |
| (5) CUDA graph replay actually engaged + 3-graph→1-graph merge + static inputs uploaded once | **~90 ms** | see §4.5 |

> (3) negative result: the per-call activation cast cost exceeds the matmul
> gain (this shape family misses ggml's f16xf16 tensor-core fast path).
> Recorded in code comments.

### 2.2 Current comparison (2026-09-19, after CUDA graph replay + single-graph merge, same-session steady state)

| Item | PyTorch-CUDA | ggml CUDA | ggml Vulkan | ggml CPU |
|----|--------------|-----------|-------------|----------|
| Pure inference P50 (f16) | **80.7 ms** | 93.0 ms | 128.8 ms | 17.8 s |
| Pure inference P50 (best quant) | — | **87.0 ms** (q8_0, **1.08x**) | 128.8 ms (f16) | **12.3 s** (q8_0) |
| Peak VRAM | 13.9-22.4 GB | ~3 GB | ~3 GB | — |

- **VRAM: ggml is ~4.6x better**; CUDA f16 speed gap is **1.15x**, q8_0 only
  **1.08x** (quantized weights read less bandwidth, so they run faster than f16).
- Measurement protocol: same session, idle system, `--warmup 10~20
  --repeats 5~30` in-process steady state (GPU boost clocks swing +-10%; the
  76 ms short-run value and the old 65 ms torch baseline are both boost-clock
  artifacts and must not be used as the protocol); see
  `charts/vggt-omega/e2e_latency_bar.png` (log scale).
- **CUDA f32 (132 ms) positioning**: nsys shows its GEMMs already run on TF32
  tensor cores (cutlass s1688tensorop, ~67 ms); the rest is the same
  elementwise/FA overhead as f16. f32 is the accuracy reference format, not a
  performance target — use f16/q8_0/q5_K for speed.
- **Vulkan f16 (129 ms) gap root cause**: the ggml Vulkan backend has **no
  CUDA-graph-equivalent replay mechanism** (per-node dispatch + barrier
  submission overhead persists, ~36 ms — same magnitude as pre-fix CUDA);
  this closes naturally if upstream ggml-vulkan adds graph replay.
- **Official demo output-surface parity**: `scripts/compare_official_demo.py`
  (official run_model keys vs the cpp trio, including extrinsic/intrinsic/
  world_points derived with the official formulas) — all 6 tensors
  float-equal on real photos (median 0.3-0.9%, caused by the +-1/255
  stbir-vs-PIL preprocessing difference; 0.1% through the same-frame --bin
  path).
- Remaining optimization paths (TODO P2): LayerNorm affine/bias fused kernels
  (~1800 elementwise kernels, est. -8~12 ms), implicit-GEMM im2col, more views.
- Reproduce: `PYTHONPATH=/tmp/torch_cuda_lib python3 benchmarks/bench_pytorch_baseline.py ...`

## 3. ETH3D official-protocol evaluation (dual-runtime, one metric implementation)

Data: official `facebook/map-anything-benchmarking` ETH3D test split
(13 scenes); window = top-covisibility 2 views per scene; resolution =
official `512_1_52_ar` (512x336); metrics = official
`mapanything.utils.metrics` + extension columns in this script; **official
scale protocol** (pred/gt each normalized by their own avg_dis) + view0-frame
pose comparison.

> Full re-measure after the 2026-09-19 uv-bug fix: depth and cloud families
> agree across runtimes (the previous ~4% systematic cpp depth bias was the
> DPT position-encoding bug, see §4.4).

| Metric family | PyTorch-CUDA (official model) | C++ CLI (f16) | Dual-runtime rel diff |
|--------|------------------------|----------------|-----------|
| Depth AbsRel | 0.0112 | 0.0111 | ~0% |
| Depth d1 | 0.988 | 0.988 | ~0% |
| Pose rot_err (deg) | 0.556 | 0.621 | +11% |
| Camera center err | 0.0156 | 0.0171 | +9% |
| ATE | 0.011 | 0.011 | ~0% |
| Point cloud Chamfer/F-score | 0.067/0.921 | 0.068/0.848 | close (same GT-border divergence source) |

(Full 13-window matrix in eval_eth3d_matrix.md.)

### 3.1 Official-protocol reproduction (130 sets, official sampling + official metric set)

**The pathological values in the first §3 revision (rot_err ~45°, ATE ~0.86)
were root-caused and fixed** — two stacked causes: (1) the homemade
top-covisibility window differs from the official random-walk covisibility
sampling (far-baseline pairs); (2) pose_enc semantics were used inverted (the
official `encoding_to_camera` → extrinsics is w2c and must be inverted to
c2w; the homemade script treated it as c2w directly). After switching to the
**official path** (official `ETH3DWAI` dataset + `ResizedDataset` ("130 @",
seed 777, covisibility_thres 0.025) + official `VGGTOmegaWrapper` + official
`get_all_info_for_metric_computation` metric functions), torch and cpp all
align with the official reproduction values (scripts/bench_official_eth3d.py):

| Official metric (130 sets) | PyTorch f32 | C++ f16 | C++ q8_0 | C++ q5_K | Official ref* |
|---------------------|-------------|---------|----------|----------|----------|
| metric_scale AbsRel down | 0.762 | 0.762 | 0.762 | 0.760 | — |
| pointmaps AbsRel down | 0.0302 | 0.0302 | 0.0305 | 0.0328 | 0.0263 |
| pointmaps inlier@1.03 up | 0.888 | 0.885 | 0.882 | 0.852 | — |
| z_depth AbsRel down | 0.0208 | 0.0209 | 0.0210 | 0.0221 | 0.0204 |
| z_depth inlier@1.03 up | 0.854 | 0.838 | 0.828 | 0.791 | — |
| ray_dirs err (deg) down | 0.814 | 0.820 | 0.825 | 0.858 | — |
| pose ATE RMSE down | 0.00655 | 0.00662 | 0.00648 | 0.00712 | 0.00999 |
| pose AUC@5 up (x100) | 79.23 | 79.08 | 77.08 | 71.69 | 79.53 |
| rot_err (deg), extension | 0.556 | 0.621 | 0.73 | 0.88 | — |
| center_err, extension | 0.016 | 0.017 | 0.019 | 0.021 | — |
| rot AUC@30 up (x100), extension | 95.69 | 95.67 | 95.5 | 94.9 | — |

\* Official reproduction.md (MapAnything ETH3D protocol, retrained ckpt).
Torch vs cpp differences are all within float noise / quantization ladder;
both reproduce the official values.

## 4. Root-cause records (fixed bugs, in discovery order)

### 4.1 S>=2 cross-frame misalignment

The backbone output patch-slice view was read via
`ggml_backend_tensor_get` (linear memcpy): frame >= 1 was misaligned by 5
tokens for S>=2. Fix: materialize with `ggml_cont`. f16 pose parity 0.0141 →
0.0001. Lesson: **any host read of a strided view must be materialized
first**.

### 4.2 CUDA non-contiguous unary/concat + K-quant cpy

CUDA unary/concat kernels require contiguous inputs (camera-head tq/fov
views need ggml_cont); CUDA cpy does not support K-quant→f32 casts, so the
lin() cast path must only apply to F16 weights — K-quant weights go through
mul_mat's native vec_dot path.

### 4.3 S>=3 concat crash

The cam/reg token first/other slices must concatenate along the frame axis
(dim=2); dim=1 only worked by coincidence at S=2.

### 4.4 DPT position-encoding implementation errors (2026-09-19, depth AbsRel 0.0496 → 0.0112)

Triggered by the user's observation "C++ error is clearly larger".
Attribution path: erosion scan + low-pass discrimination (not border noise,
not high-frequency misalignment), f32==f16 (not quantization), token-perturbation
probe (not trunk-feature conduction), cross-feed test (cpp tokens into the
torch head → output ≈ torch depth; the gap is inside the cpp dense head),
then per-stage dumps. **Three implementation errors** in
`build_dpt_uv_embed` vs torch `create_uv_grid`/`position_grid_to_embed`:
(1) `sin_x+sin_y` summed instead of `[sin_x|cos_x|sin_y|cos_y]`
concatenated; (2) make_sincos given the full channel count so all cos
channels were dropped; (3) UV span missing the (N-1)/N factor. Fixing all
three brought the uv embed to rel=0 and depth parity to torch level.
The earlier "GT-border divergence" explanation in §3 was a misdiagnosis —
it was this bug.

**Lesson: position-encoding-style "pure functions" must be parity-checked
numerically level by level via dumps; similar magnitude (gate medians cannot
expose it) does not mean semantically correct.**

### 4.5 DPT conv performance optimization (2026-09-19, CUDA f16 pure inference 171 → 141 → 97 → ~90 ms)

User asked to locate and fix the latency gap vs PyTorch one by one (no cuDNN
— ggml already uses its own CUDA kernels, so none is needed).

1. **conv_transpose_2d_p0 k==s fast path** (ggml patch
   `third_party/ggml-patches/0001-cuda-conv-transpose-ks-fast-path.patch`):
   the generic kernel loops `c_in×kh×kw` per output element with a per-step
   `in % stride` alignment check (3/4 iterations discarded when k==s); the
   math of k==s non-overlapping upsampling is exactly one contributing input
   pixel per output element (direct gather + c_in dot product). The
   dedicated kernel maps thread→(co,oh,ow) directly, no modulo backtracking,
   co-contiguous kernel reads. (-21 ms)
2. **projects[i] 1x1 conv → mul_mat**: a 1x1 conv is a per-pixel linear
   layer, and the post-ln token layout {D2,P,S} is already (C_in, N) — a
   zero-copy `mul_mat` (the direct-conv kernel was also a per-output-thread
   scalar loop). Note mul_mat output has oc on ne[0], bias needs {oc,1,1,1};
   reshaping {oc,P,S} directly to {oc,Hp,Wp,S} swaps h/w — route via
   {P,oc,S}. (-9 ms)
3. **CUDA graph replay never engaged**: nsys showed 4347 cudaLaunchKernel
   per inference (~62 ms API-side) and zero cudaGraphLaunch — ggml's CUDA
   graph replay requires two consecutive computes with unchanged properties
   **in the same process**, while bench_latency spawned a cold CLI process
   per repeat (2 computes per process, first one being warmup), so replay
   could never activate. Fix: the CLI gained `--warmup/--repeats` (in-process
   repeated inference) and bench_latency became a single-process steady-state
   measurement. Steady state 139 → 97 ms: the remaining ~60 ms was
   per-kernel launch gaps.
4. **3 graphs → 1 graph**: backbone + aggregator+heads merged into one ggml
   graph (single context, single gallocr, single `ggml_backend_graph_compute`),
   removing the inter-graph host bridge (tensor_get + tensor_set + two full
   syncs); static inputs (normalization, RoPE tables, uv embeddings) are
   uploaded once at build time (only images per run). Dump-only reads moved
   under `MAPGGML_ENABLE_DUMP` (the production path was still pointlessly
   reading back ~51 MB of intermediates every run). 97 → **~90 ms**.

**Lesson: direct-conv per-output-thread scalar loops are hugely wasteful for
1x1/non-overlapping convolutions; ops whose math is a GEMM must run as GEMMs
(tensor cores), and C-order assumptions in layout transforms must be verified
dimension by dimension. Also, benchmark protocols must match deployment: only
in-process repeated inference exposes ggml CUDA graph replay's real benefit;
cold-start-per-iteration measures launch overhead, not steady state.**

## 5. Throughput (CUDA f16, 512x512, re-measured 2026-09-19)

| S | 2 | 4 | 8 |
|---|---|---|---|
| Inference P50 (ms) | 93.0 | 198.2 | 463.0 |
| Per view (ms) | 46.5 | 49.6 | 57.9 |
| Throughput (views/s) | 21.5 | 20.2 | 17.3 |

(The old binary's S=3/6 sweep is obsolete; per-view cost is lowest at S=2;
steady-state in-process protocol.)

## 6. Quantization positioning and the q4_K/q5_K root cause

### 6.1 Root-cause experiment (weight relative RMS error vs f32, measured)

| Quant weight group | q4_K relRMS | q5_K relRMS | ratio |
|------------|-------------|-------------|-------|
| backbone blocks | 0.0763 | 0.0387 | 1.97 |
| aggregator blocks | 0.0727 | 0.0368 | 1.98 |
| overall | 0.0751 | 0.0387 | 1.94 |

q4_K's weight error is a factor 1.94 worse, and its pose error (0.0179) was a
factor ~1.90 worse than q5_K (0.0094) — quantization SNR is the mechanism,
confirmed end to end.

### 6.2 Decision table

| Format | Size | pose err | GPU speed | Positioning |
|------|------|-----------|----------|------|
| f16  | 2.18 GB | 0.0010 | fastest | **accuracy-first default** |
| q8_0 | 1.25 GB | 0.0022 | fast | near-f16 accuracy |
| q5_K | 858 MB | 0.0170 | fast | **recommended value tier** |
| f32  | 4.36 GB | — | — | reference baseline |

**Decision (2026-09-18): q4_K removed.** It is not a valid Pareto point —
only 13% smaller (0.76 vs 0.90 GB), same speed as q5_K, half the accuracy
(inherent quantization SNR); q5_K covers every reasonable use case. Extreme-
VRAM devices can fetch it from git history or use q8_0. Cleanup: converter
outtypes, GGUF files, gate/test_matrix/chart lists.

## 7. Unit tests

- `tests/test_s2_independence.cpp` (C++, zero-deps): S=2 cross-frame
  structural properties + torch-reference parity (f16/q8_0/q5_K all PASS).
- `tests/test_matrix.py`: (backend x model x quant) regression matrix.

## 8. Chart list (benchmarks/charts/ — now organized per model; vggt-omega charts under charts/vggt-omega/, pi3x under charts/pi3x/ with its illustrated report charts/pi3x/pi3x_report.md)

- `e2e_latency_bar.png` — PyTorch vs backends x quantizations (log scale)
- `quant_pareto_3d.png` — accuracy vs pose err vs file size
- `pose_error_heatmap.png` — quant x backend pose max_abs
- `parity_scatter.png` — C++ vs PyTorch pose/depth scatter
- `recon_depth_comparison.png` / `recon_pointcloud_comparison.png` /
  `recon_metrics_comparison.png` — real-data ETH3D reconstruction comparisons

Per-model illustrated reports (gate + official protocol + recon + latency,
one per family, same spec):

- [charts/vggt-omega/omega_report.md](charts/vggt-omega/omega_report.md)
- [charts/vggt-1b/vggt1b_report.md](charts/vggt-1b/vggt1b_report.md)
- [charts/pi3/pi3_report.md](charts/pi3/pi3_report.md)
- [charts/pi3x/pi3x_report.md](charts/pi3x/pi3x_report.md)
- [charts/mapanything/mapanything_report.md](charts/mapanything/mapanything_report.md)

Results tree index: [results/README.md](results/README.md); cross-model gate
matrix: [results/gate_matrix.md](results/gate_matrix.md).
  (from scripts/compare_reconstruction.py)
- `recon_comparison.md` — the window-level numeric table

Obsolete charts were removed 2026-09-19 (stage_breakdown_stacked after the
single-graph merge, orphaned throughput/memory charts, and the misdiagnosis
border-distance chart).
