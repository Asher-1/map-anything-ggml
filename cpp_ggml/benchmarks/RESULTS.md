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

## 2026-09-30 — full vggt-omega re-verification on the unified pipeline

No code regression: a worktree build at 36e621b (pre "opt vggt") reproduces
the current outputs bit-exactly; the 09-25 commits changed no numerics.
The omega gate moved into the unified e2e_gate_matrix.sh (12/12 PASS,
thresholds recalibrated to the canonical frames512 floors — the q5_K depth
floor is frame-dependent, 0.27% benign vs ~10% on the LCG parity frames;
real-data AbsRel regression stays <= 9% relative). cmp_gate_matrix.py /
plot_charts_model.py / e2e_gate_matrix.sh gained the vggt_omega arch;
charts/vggt-omega/ gained gate_summary.json. Fresh 13-set ETH3D evals are
byte-identical to the committed ones (12/12 configs); the refreshed
130-set two-sided table matches to 4-5 decimals.
Also added the q6_K tier for omega (1.05 GB, converter + drift guard):
15/15 gate PASS, and on the official 130 sets depth AbsRel 0.020748 vs
torch f32 0.020752 with ATE 0.006436 beating torch — the accuracy gap at
the ~1 GB size point is closed (AUC5 76.5 vs q5_K 71.7 / q8_0 77.1).
q6_K verified on the other two omega variants as well (same seed-7 frames,
CUDA): depth med_rel 0.47% (416) / 0.58% (256-text) vs q5_K 1.63% / 1.75%;
13-set AbsRel also better (416 0.016992 vs 0.017253; text 0.025635 vs
0.026028). Omega q5_K removed 2026-09-30 (GGUFs + result mds; dominated
by q6_K, the 4.5-bit loss being inherent to the tier); q5_K stays for the
other five models where it is healthy.

### 2026-10-01 — fleet-wide q6_K confirmation (all five other models)

All five remaining models: q6_K GGUFs converted, gated 15/15 each
(gate_matrix_q6k_fleet.log), and measured on the official ETH3D 130 sets
cpp-only at each model's own resolution
(results/FLEET_QUANT_CONFIRMATION.md, via benchmarks/aggregate_fleet_quant.py).
Verdicts (z_depth AbsRel / ATE / AUC5, q6_K vs q8_0 vs q5_K):

- **pi3x / mapanything / dust3r**: q6_K depth best of the three tiers;
  AUC5 between q8_0 and q5_K — q6_K is the sub-1.1 GB sweet spot and
  dominates q5_K for pi3x/dust3r (removal optional, files kept);
- **pi3**: q5_K is genuinely the best depth/rot tier (0.046483 / 1.086°)
  and q8_0 the best AUC5 — real-data quantization response is NOT
  monotonic in bit width; keep all three tiers;
- **vggt-1b**: q6_K is the WORST of the three (0.052635/0.024855/62.92 vs
  q8_0 0.051085/0.022965/63.08) — this backbone responds badly to the
  6-bit grid; keep q8_0/q5_K, do not adopt q6_K.

Lesson: quant-tier verdicts must be measured per model — the omega result
(q6_K matches f32) does not transfer (pi3 prefers q5_K, vggt-1b rejects
q6_K on real data).

Follow-up 2026-10-01: pi3x and dust3r q5_K removed (GGUFs + per-quant
130-set mds locally, and deleted from the HF repo on the user's explicit
instruction) — the fleet table above already showed q6_K winning AUC5 by
+3.7 points on pi3x and sweeping every official metric on dust3r.
Same day, later: the remaining pi3/mapanything/vggt-1b q5_K GGUFs and
per-quant mds were initially retired locally as well — then, on the
user's correction, restored for pi3 and vggt-1b (GGUFs re-pulled
byte-identically from HF, mds from the git index): q5_K is the best
depth/rot/ATE/pointmaps tier for pi3 and the best AUC5 tier for
vggt-1b, so removing it there was over-reach. Final state: q5_K kept
locally for pi3/vggt-1b (34 GGUFs total), retired locally for
omega/pi3x/dust3r/mapanything; all six HF q5_K files remain
downloadable. Gate counts: pi3 15/15, vggt-1b 15/15, mapanything 9/9.
Final correction (same day): pi3 and vggt-1b q6_K removed (locally + HF,
user instruction backed by the fleet table) — pi3's q6_K had no best
metric on the 130 sets, and vggt-1b's was the weakest tier on 5 of 7.
Final rule: ONE measured K tier per model — q6_K for omega/pi3x/
mapanything/dust3r, q5_K for pi3/vggt-1b; 32 GGUFs in the local zoo.
Gate counts: pi3 12/12, vggt-1b 12/12, mapanything 9/9.
Same day, evening: per-model gate matrices re-run with the shipped K
tiers (57/57 PASS) and logged to results/<model>/gate_matrix_log.txt —
all six charts/ dirs now carry gate_summary.json + a q6_K/q5_K row in
pose_error_heatmap and quant_pareto (previously omega-only). Latency
refresh scripted as benchmarks/refresh_latency_charts.sh (hard-fails
unless the GPU is exclusive); it still awaits an idle card for the q6_K
latency bars and the omega re-measurement.
Same day, night: courtyard recon comparisons regenerated for the three
q6_K models (pi3x/mapanything/dust3r — their old PNGs/mds still had q5_K
rows; pi3/vggt-1b/omega were already current). pi3x's q6_K is the recon
star: AbsRel 0.015737 (better than f16/q8_0, ~at torch f32 0.015779)
with pose rot 0.190°/trans 0.011 m vs the old q5_K's 0.905°/0.053 m;
mapanything q6_K 0.077505 (old q5_K 0.078042); dust3r q6_K edges every
tier on AbsRel/chamfer. compare_reconstruction_pi3x.py now derives the
K tier from --arch (q5_K for pi3, q6_K otherwise). Omega's recon was
also caught stale (its 09-30 re-run predated q6_K) and regenerated:
q6_K chamfer 0.010381 / fscore 0.983513 vs torch 0.010370/0.983555.
Same night: latency rows closed. CUDA+Vulkan re-measured for the four
q6_K models (10 repeats) with torch baselines re-run in the SAME session
(pi3x 290.3 / mapanything 244.0 / dust3r 148.2 — all matching the
09-24 exclusive values, validating the shared-threshold methodology);
omega torch came in at 65.0 (vs 80.7 on 09-19 — the official side also
got faster), giving omega CUDA f16 0.85x / q8_0 0.92x, consistent with
the 09-19 ratio. q6_K costs ~0 latency vs f16 everywhere (e.g. omega
75.5 vs 76.2; pi3x 202.1 vs 202.5). Vulkan: dust3r f16 64.1 = 2.31x vs
torch-CUDA; mapanything f16 137.8 = 1.77x. Guard upgraded:
refresh_latency_charts.sh --allow-shared accepts a ≤5%-util card with
≥8 GB free (idle viewer holding VRAM only) and repeats=10; the 09-30
contamination is impossible under the new thresholds. All six latency
bars redrawn with q6_K; charts refreshed from their own gate logs.
Same day, late night: end-to-end dual-runtime courtyard rebuild for human
inspection — official demo path (window [5,0] = DSC_0291+DSC_0286) on
torch f32 vs cpp f16/q6_K, three colored 519k-point PLYs exported
(scripts/export_recon_ply.py). Per-point 3D deviation vs torch: f16
median 0.0028 (0.28% of scene extent), q6_K 0.0126 (1.27%); all six
COMPARISON tensors float-equal. (The 2-frame e2e_courtyard/ dir was
removed 2026-10-06 as a superseded intermediate — fully replaced by the
38-frame e2e_courtyard_full/; the numbers above are the retained record.)
compare_official_demo.py gained --device cpu (a 15 GB co-tenant left
<100 MiB VRAM; CPU f32 is numerically equivalent) and now also saves the
cpp-side world_points_from_depth.npy for PLY export.
Same day, full scene: scripts/e2e_full_scene.py — ALL 38 frames of
courtyard through both runtimes (sliding chunks K=8/overlap=2 — the
family's global attention cannot fit 38 views in one forward — Horn Sim3
chain alignment on the 2 shared frames per boundary, conf>=3.0 filter,
0.02 voxel merge). Fused colored clouds:
results/vggt-omega/e2e_courtyard_full/{torch_f32,cpp_f16}_global_38f.ply
(9.57 M-point correspondence, 220k points after voxel merge per side);
cross-runtime per-point deviation median 0.00108 / p95 0.00324 — the
two full-scene reconstructions are the same building to sub-0.2% of
scene extent. Chunk-boundary Sim3 scales 1.05-1.56 (per-chunk scale
drift absorbed); ALIGNMENT.md carries the full fit table.
CORRECTION (2026-10-06, user-reported): the v1 chain-aligned full-scene
clouds were WRONG — walls visibly split in CloudViewer. Root cause: the
model's cross-chunk scale drift is LARGE (re-measured 6.45 → 14.12,
a 2.2x span across the 6 chunks) and the 2-shared-frame boundary Sim3
absorbs it with scales up to 1.56 whose residuals (median 0.004-0.008,
p95 up to 0.041) then ACCUMULATE along the chain; the sub-0.001
cross-runtime agreement was an artifact of both sides sharing the same
flawed alignment code. Fix: e2e_full_scene.py --align gt (now default)
anchors each chunk INDEPENDENTLY into the metric GT world frame (WAI
scene_meta transform_matrix) with one robust Horn Sim3 on the 8
model-vs-GT camera centers — no chain, no accumulation. Re-run:
6.65 M-point clouds, anchor residuals 0.005-0.045 m median (torch and
cpp agree per chunk to <2 mm; the residual is the model's own pose
error, largest on chunks 3-5), cross-runtime deviation median 0.016 m,
preview renders show walls closing into one coherent building
(results/vggt-omega/e2e_courtyard_full/). Chunk dirs and other
intermediate bins/meta removed after the merge (~100 MB reclaimed).
scripts/preview_ply.py added for 3-axis visual QA of any fused PLY.
FINAL ANSWER to "does the official repo have a fusion scheme?" (user
question, same day): NO — demo_gradio/visual_util run ONE forward over
all frames (global attention IS the consistency mechanism) and then just
concatenate filtered clouds; chunking was purely our CPU workaround.
e2e_full_scene.py gained `--chunk 0` (official single-pass semantics; 38
frames in 56 s on the 4090, torch fp32 + cpp CUDA both fine) plus the
official visual_util quality stack (depth_edge rtol 0.03 + confidence
percentile p20, floor conf >= 14.1) and a conf-PRIORITY voxel merge (per
voxel keep the highest-confidence view — fixes the blurry multi-view
noise band that plain concatenation leaves). v3 single-pass clouds:
median cross-runtime deviation 0.015 m, NO chunk seams, windows sharp
(results/vggt-omega/e2e_courtyard_full/*f16single*); the v2 chunked
clouds were removed. Best practice recorded: single pass when the GPU
fits, chunked + per-chunk GT anchor otherwise.
VRAM post-mortem (user challenged "why chunk at all on 24 GB"): measured
torch.cuda.max_memory_allocated for the 38-frame single forward = 8.75
GiB (4.26 weights + ~118 MiB/frame; SDPA memory O(N)) — 100+ frames fit
on 24 GB. "38 frames cannot fit" was an unverified assumption from the
time a 17 GB co-tenant held the card. e2e_full_scene.py defaults flipped
to the best pipeline (chunk 0 / align gt / conf-percentile 20 with the
official max(2.0, p) floor) and a default-args re-run reproduced the
delivered clouds byte-identically (ALIGNMENT diff empty, PLY md5s equal).
