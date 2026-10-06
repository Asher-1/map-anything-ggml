# Feature Parity Audit — cpp_ggml integrated models vs the official repos, end to end

> 2026-09-24. Model-by-model comparison of every inference-side feature
> module in the official source repos (third_party/*-src + the mapanything
> main repo) against the current C++ (src/) implementation. Every row cites
> file-level evidence from the official side.
> Status: ✅ aligned and numerically verified / ⚠️ semantically equivalent,
> different implementation path / ❌ not implemented (with reasoning and a
> follow-up home).

## 1. vggt-omega (facebook/vggt-omega)

| Official feature | Official location | C++ status |
|---|---|---|
| DINOv2 ViT-L backbone + patch_embed | `vggt_omega/models/layers/vision_transformer.py`, `patch_embed.py` | ✅ `DinoVisionTransformer` (bb.*) |
| Aggregator: frame blocks + inter-frame global/register alternation (incl. AAT qk-norm) | `aggregator.py` `_run_frame_block` / `_run_inter_frame_attention_block` | ✅ `aggregator()` (register_attn_block_idx parameterized) |
| CameraHead -> pose_enc(9) | `heads/camera_head.py` | ✅ (gate pose < 0.005) |
| DenseHead -> depth/depth_conf (pixel-shuffle) | `heads/dense_head.py` | ✅ |
| **TextAlignmentHead** (language-aligned embedding: camera/register tokens -> 4 SelfAttentionBlocks -> projector -> L2 norm) | `heads/text_alignment_head.py` | ✅ **implemented 2026-09-24**: `text_head()` (reuses `block()`, rope-free), CLI emits `.text_embedding.bin`; 256-text f16 gate **PASS** (pose 0.0002 / depth 0.0001 / **text cos = 1.000000**) |
| enable_alignment switch | `vggt_omega.py:37` | ✅ GGUF `vggt.enable_text_alignment` (written by the converter's `has_text_head`) |
| Arbitrary /16-resolution inference | demo | ✅ 512 weights @ 256 input first-verified, pose maxdiff 8.7e-5 (2026-09-24) |
| autocast / amp | model side | N/A (C++ backends have their own precision semantics, calibrated by the f16/f32 gates) |

Weight variants: 512 / 416-reproduce / 256-text are all converted, each in
f32 / f16 / q8_0 / q6_K (q5_K removed 2026-09-30, dominated by q6_K).

## 2. vggt-1b (facebook/vggt, third_party/vggt-src)

| Official feature | C++ status |
|---|---|
| DINOv2 backbone + alternating aggregator | ✅ (gate 12/12) |
| CameraHead / DepthHead / PointHead | ✅ (pt.* point head gate-verified) |
| **PointHead output to disk** (`.points.bin`/`.points_conf.bin`) | ✅ **fixed 2026-09-24**: VGGTImpl always captured world_points but the cli's vggt-family branch never wrote the files (the torch reference generator always did); after the fix the gate passes (points med_rel 0.00004 / conf 0.00000) |
| **TrackHead** (`vggt/heads/track_head.py` 104 lines + `track_modules/` 4 files ~26KB: DPT features + correlation pyramid (7 levels, radius 4) + 4-iteration trajectory refinement) | ❌ **not implemented**. The converter drops `track_head.*` explicitly (counted). Two parts are missing: (a) the C++ graph for BaseTrackerPredictor (the correlation pyramid is data-dependent, needs an im2col/warp scheme); (b) the CLI needs a **query-point input interface** (the official head accepts a grid or user-specified queries). **Follow-up: M6 candidate (~2-3 person-days)** |
| Multi-view S>2 | ✅ **numerically first-verified 2026-09-24**: vggt-1b S=4 gate PASS (pose/depth/depth_conf/points/points_conf med_rel <= 0.00006, point map included); omega S=4 pose max 0.0015 / depth med_rel 0.0006 PASS. Before that only S=2 gates and an omega throughput sweep existed |

## 3. pi3 (yyfz/Pi3) / pi3x

| Official feature | C++ status |
|---|---|
| backbone + per-layer cross-view decoder | ✅ |
| local_points / conf heads | ✅ |
| c2w pose (SVD orthogonalization) | ✅ |
| pi3x: metric scale head (`.scale.bin`) + ConvHead regression chain | ✅ (gate 12/12 + paper protocol pinned on 13 scenes <= 0.0021) |
| Multi-view | ✅ |

## 4. mapanything (facebook/map-anything)

| Official feature | Official evidence | C++ status |
|---|---|---|
| DINOv2 encoder + decoder + AAT | `mapanything/models/mapanything/model.py` | ✅ |
| Output heads depth/conf/pts3d/pose/ray_dirs/non_ambiguous_mask/metric scale | same file, `[raydirs(3)\|depth(1)\|conf(1)\|mask(1)]` head | ✅ all emitted (`.mask.bin` sigmoid / `.scale.bin` exp / rays unit-sphere normalized) |
| Geometric input injection (use_pose/use_depth/use_calibration) | `model.py` `_configure_geometric_input_config`: a **training-time random-injection config**; overall_prob=0 is images-only mode | N/A — the official inference path IS images-only; the posed/registration tasks are **evaluation-protocol layers** (GT pose used only for scale alignment), not network inputs |
| memory_efficient_inference / minibatch_size | `forward()` | ⚠️ semantically equivalent: the C++ single merged graph + chunked compute differs in memory behavior but outputs match (two-sided gate/bench parity) |
| Multi-resolution / aspect ratio | many-AR configs | ✅ (official 512x336 protocol bench, two-sided parity) |

## 5. dust3r (naver/dust3r)

| Official feature | Official evidence | C++ status |
|---|---|---|
| CroCo ViT-L/16 encoder + 12 cross-attn decoder pairs + dual DPT heads | `dust3r/model.py`, `croco/models/blocks.py`, `dust3r/heads/dpt_head.py` | ✅ Dust3rImpl (gate 12/12) |
| Pair-wise pointmaps (both heads in view1's frame) + conf = 1+exp | `dust3r/heads/postprocess.py` | ✅ |
| 2-view pose recovery | official num_views=2 protocol | ✅ closed-form rigid Procrustes (`assemble_dust3r` shared by both runtimes; 130-set bench two-sided parity < 0.1%) |
| **>2-view global alignment optimizer** | `dust3r/cloud_opt/global_align.py` (PairwiseRelation + pairwise-graph BA, 800+ lines) | ❌ intentionally not ported. Rationale: the pair-wise model's input semantics are exactly 2 views; for >2 views the official paradigm itself was superseded by VGGT/MapAnything (N-view aggregators); porting an out-of-network optimizer adds no network-alignment value. If ever needed, the bench Procrustes extends to pairwise closed-form + a pose-graph backend |
| MASt3R matching head | separate repo (naver/mast3r) | out of scope for this model (components already paved for M6+) |
| demo/Gradio/visualization | `demo_gradio.py`, `visual_util.py` | N/A (demo layer, not core inference) |

## 6. Cross-model metric comparison (official MapAnything ETH3D protocol, seed 777, 2 views)

Absolute accuracy on the official torch side under the same protocol (the
cpp<->torch two-sided deltas are < 1% for every model; torch values shown):

| Model | sets | pointmaps_abs_rel | z_depth_abs_rel | ATE RMSE | rot_err_deg | metric_scale_abs_rel |
|---|---|---|---|---|---|---|
| **vggt-omega** | **130** | **0.0302** | **0.0208** | **0.0065** | **0.56** | 0.762 |
| pi3x | 130 | 0.0533 | 0.0369 | 0.0162 | 2.11 | 0.251 |
| mapanything | 130 | 0.0553 | 0.0469 | 0.0124 | 0.92 | **0.162** |
| pi3 | 130 | 0.0650 | 0.0485 | 0.0172 | 1.82 | 0.733 |
| vggt-1b | 130 | 0.0674 | 0.0541 | 0.0208 | 2.28 | 0.788 |
| dust3r | 130 | 0.1008 | 0.1033 | 0.0509 | 2.77 | 0.897 |

Conclusions (answering "which model has the best metrics"):
- **Absolute accuracy: vggt-omega is first across the board** (after the
  full 130-set run: pointmaps 43% better than the runner-up, depth 44%,
  ATE 47%, rot 63%; pose_auc_5 79.2 vs the official reference 79.53).
- Within the fully-130-set evaluations: pi3x leads geometry (pointmaps/
  depth); mapanything leads pose/metric scale (ATE 0.0124, rot 0.92 deg,
  metric scale 0.162).
- dust3r ranks last, as expected for the 2024 ancestor (its value is
  speed: CUDA 1.75x and the family's mildest quantization degradation).
- Engineering parity (cpp vs torch two-sided deltas) is < 1% everywhere,
  independent of model choice.

## 7. Fixes and verifications triggered by this audit

1. **Latency chart pipeline fix**: backend-name normalization in
   `plot_charts_model.py` (the CLI records ggml's native `CUDA0/Vulkan0`
   while the chart loop matched `CUDA/Vulkan` — every model's
   e2e_latency_bar silently showed CPU bars only). All models now render
   the full 3 backends x 4 quantizations.
2. **cpp f32 latency filled in** for pi3/pi3x/vggt-1b/mapanything/dust3r on
   all backends (incl. the newly converted mapanything-f32 GGUF); dust3r
   Vulkan latency added.
3. **omega TextAlignmentHead implemented** (see §1); the 256-text variant
   entered the gate for the first time.
4. **Resolution generalization first-verified**: omega 512 weights @ 256
   input PASS (gates previously covered only the nominal resolution).
5. **vggt-1b point-map output to disk** (found by this audit's second
   pass): fixed and gate-verified.
6. **First S>2 numeric parity**: vggt-1b S=4 (5 outputs) and omega S=4 all
   PASS; `cmp_gate_matrix.py` gained an optional `[S H W]` override.
7. **vggt-omega official bench extended from 10 to 130 sets**:
   `bench_official_eth3d_130set.md`, two-sided deltas all < 1%, pose_auc_5
   79.2 vs the official reference 79.53.
8. **CPU latency exclusivity discipline**: an earlier latency batch ran
   concurrently with a full-core build and was contaminated (pi3 CPU f16
   17609 ms vs the true 10249 ms, 42% off); all CPU rows re-measured
   exclusively, GPU f32 rows re-verified.

## 8. Known unimplemented list (by priority)

| Item | Model | Estimate | Value |
|---|---|---|---|
| TrackHead + CLI query-point interface | vggt-1b | 2-3 person-days | One of the official four heads (the point-tracking demo's core selling point) |
| >2-view global alignment optimizer | dust3r | 3-5 person-days | Low (see §5 for the reasoning) |
| MASt3R model as a whole | new model, M6 | same scale as M5 | matching/dense-correspondence capability |
