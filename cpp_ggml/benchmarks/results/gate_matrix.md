# (arch x quant x backend) parity gate matrix — 2026-09-21

Fixed torch-f32 references, 518x518 S=2 (frames518.bin):
pi3 <- spy_torch_pi3.py, mapanything <- spy_torch_mapanything.py,
vggt <- dump_torch_vggt.py. Gate thresholds scale with quantization
(f32/f16 x1, q8_0 x3, q5_K x6). Result: **33/33 PASS**.

## pi3 (worst-output median_rel vs torch f32)

| quant | CPU | CUDA | Vulkan |
|---|---|---|---|
| f16 | pose 0.00050 / lp 0.00363 / conf 0.0236 | 0.00066 / 0.00308 / 0.0228 | 0.00035 / 0.00268 / 0.0206 |
| f32 | 0.00051 / 0.00376 / 0.0242 | 0.00072 / 0.00341 / 0.0249 | 0.00036 / 0.00268 / 0.0206 |
| q8_0 | 0.00185 / 0.0169 / 0.0739 | 0.00174 / 0.0147 / 0.0728 | 0.00123 / 0.0110 / 0.0520 |
| q5_K | 0.00383 / 0.0159 / 0.0430 | 0.00389 / 0.0123 / 0.0855 | 0.00380 / 0.0107 / 0.0544 |

(conf is raw logits with |ref| ~ 1-10, so its med_rel overstates; the
f16 CPU numbers equal the recorded e2e gate 0.00248/0.00364/0.02266.)

## mapanything (9/9 PASS; worst-output median_rel)

| quant | CPU | CUDA | Vulkan |
|---|---|---|---|
| f16 | 0.00015 | 0.00167 | 0.00128 |
| q8_0 | 0.00212 | 0.00105 | 0.00138 |
| q5_K | 0.00498 | 0.00539 | 0.00485 |

## vggt (worst-output median_rel)

| quant | CPU | CUDA | Vulkan |
|---|---|---|---|
| f16 | 0.00248 | ~0.0011 | ~0.0025 |
| f32 | 0.00004 | 0.00205 | 0.00248 |
| q8_0 | 0.00324 | 0.00103 | 0.00171 |
| q5_K | 0.01783 | 0.01676 | 0.01375 |

Observations:
- The three backends agree to the same order on every (arch, quant);
  CUDA/Vulkan-vs-CPU deltas are backend float-order noise, not backend bugs.
- q8_0 costs < ~0.5% typical, q5_K < ~2% typical, consistent with the
  omega q8_0/q5_K ETH3D results recorded earlier.
- vggt-1b q8_0/q5_K were REGENERATED today: the converter QTYPES table had
  a wrong q8_0 type_size (20 instead of 34) that segfaults quantization and
  would have produced truncated GGUFs; the pre-existing omega q8_0/q5_K
  files predate the bug (verified by size arithmetic) and remain valid.

## pi3x — 2026-09-22, 12/12 PASS (added to the matrix)

Gate thresholds calibrated to pi3x's own noise floor: lp med_rel
~0.015 for BOTH f16 and f32 (flash_attn_ext keeps K/V f16; the
retrained decoder amplifies it ~9x) -> base threshold 0.020.
Cells: pose16 med_rel / local_points med_rel / scale rel.

| quant | CPU | CUDA | Vulkan |
|---|---|---|---|
| f16 | pose 0.00123 / lp 0.01498 / sc 0.00036 | pose 0.00174 / lp 0.01683 / sc 0.00031 | pose 0.00199 / lp 0.01858 / sc 0.00028 |
| f32 | pose 0.00125 / lp 0.01486 / sc 0.00038 | pose 0.00184 / lp 0.01631 / sc 0.00031 | pose 0.00203 / lp 0.01859 / sc 0.00029 |
| q8_0 | pose 0.00193 / lp 0.01389 / sc 0.00033 | pose 0.00281 / lp 0.01199 / sc 0.00038 | pose 0.00292 / lp 0.02340 / sc 0.00041 |
| q5_K | pose 0.00934 / lp 0.08797 / sc 0.00338 | pose 0.00917 / lp 0.07246 / sc 0.00369 | pose 0.01073 / lp 0.08421 / sc 0.00369 |


## dust3r — 2026-09-23, 12/12 PASS (M5, added to the matrix)

Pair-wise branch (512_dpt, 512x512 S=2, teaser-frame parity). Gate
thresholds calibrated to dust3r's own noise floor: lp med_rel ~0.0002
(f16/f32, the DPT head has no amplifying res-chain like pi3x's ConvHead)
-> base threshold 0.020 left generous. Cells: local_points med_rel /
conf med_rel.

| quant | CPU | CUDA | Vulkan |
|---|---|---|---|
| f16 | lp 0.00019 / conf 0.00056 | lp 0.00153 / conf 0.00475 | lp 0.00164 / conf 0.00382 |
| f32 | lp 0.00019 / conf 0.00052 | lp 0.00141 / conf 0.00304 | lp 0.00164 / conf 0.00382 |
| q8_0 | lp 0.00155 / conf 0.00881 | lp 0.00237 / conf 0.00751 | lp 0.00202 / conf 0.00539 |
| q5_K | lp 0.00387 / conf 0.01556 | lp 0.00868 / conf 0.05333 | lp 0.00690 / conf 0.05650 |

Observations:
- Quantization degradation is remarkably mild (q8_0 ~0.2%, q5_K ~0.4-0.6%
  on lp) — the croco decoder + dust3r DPT is the most quant-robust
  architecture in the repo so far; f16 is recommended (already 2x faster
  weights than f32 at identical accuracy).
- Conf is 1+exp(x) (vmin=1): its med_rel overstates (denominator ~1).
- No pose outputs (dust3r has no pose head; global alignment is an
  out-of-network optimizer, not ported).

## vggt-omega-256-text — 2026-09-24, f16 CPU PASS (text head added)

The C++ graph now implements the official TextAlignmentHead (M-omega
follow-up; weights were already travelling in the 256-text GGUF as f32/f16
extras alongside the `vggt.enable_text_alignment` flag). 256x256 S=2
teaser-frame gate, official torch reference via `dump_torch_stages.py`
(built with enable_alignment=True):

| key | value |
|---|---|
| pose_enc max_abs | 0.00016 |
| depth median_rel | 0.0001 |
| text_embedding cosine | **1.000000** (gate > 0.999) |

Also first-verified this day: resolution generalization (512 weights @ 256
input, pose maxdiff 8.7e-5) — the gate matrix had only ever covered the
nominal per-variant resolution.

## vggt-1b points output + S=4 multi-view parity — 2026-09-24

Two residual-parity findings closed the same day:
1. `.points.bin`/`.points_conf.bin` were captured by VGGTImpl but never
   written by the cli's vggt-family branch (the torch reference generator
   always emitted them). After the cli fix the S=2 gate passes with
   points med_rel 0.00004 / points_conf 0.00000.
2. First-ever S>2 numeric parity: vggt-1b at S=4 (518x518) passes all five
   outputs (med_rel <= 0.00006); vggt-omega at S=4 passes pose max 0.0015 /
   depth med_rel 0.0006. cmp_gate_matrix.py gained an optional [S H W]
   override for non-default view counts.
Also this day: vggt-omega's official ETH3D bench extended from 10 to 130
sets (bench_official_eth3d_130set.md) — two-sided deltas all <1%,
pose_auc_5 79.2 vs the official reference 79.53.

## vggt-omega — 2026-09-30, 15/15 PASS (moved into the unified matrix)

First inclusion here (omega previously gated through scripts/e2e_gate.sh
with its fixed 0.005 thresholds). Contract: pose_enc (S,9) + depth (S,H,W)
+ depth_conf (S,H,W) at 512x512 S=2 on the canonical /tmp/frames512.bin;
torch f32 reference via dump_torch_stages.py (ref.npz). med_rel per output:

| quant | CPU | CUDA | Vulkan |
|---|---|---|---|
| f16 | pose 0.00042 / depth 0.00059 / conf 0.00064 | pose 0.00496 / depth 0.00174 / conf 0.00678 | pose 0.00377 / depth 0.00180 / conf 0.00480 |
| f32 | pose 0.00040 / depth 0.00047 / conf 0.00037 | pose 0.00031 / depth 0.00323 / conf 0.00283 | pose 0.00377 / depth 0.00180 / conf 0.00480 |
| q8_0 | pose 0.00146 / depth 0.00602 / conf 0.00771 | pose 0.00304 / depth 0.00818 / conf 0.01131 | pose 0.00311 / depth 0.00325 / conf 0.00992 |
| q6_K | pose 0.02426 / depth 0.00655 / conf 0.01274 | pose 0.02563 / depth 0.00532 / conf 0.02645 | pose 0.02459 / depth 0.00857 / conf 0.02536 |
| q5_K | pose 0.02330 / depth 0.08888 / conf 0.02094 | pose 0.01183 / depth 0.10382 / conf 0.02791 | pose 0.01029 / depth 0.04394 / conf 0.02582 |
  (q5_K removed 2026-09-30 — dominated by q6_K; row kept as the final
  measured record)

(f32 Vulkan equals the f16 row: the Vulkan path upcasts weights to f32,
like the CPU lin() cast — the f16 and f32 GGUFs are numerically the same
model on that backend.)

Thresholds calibrated to the canonical-frame floors with >= 1.4x margin:
pose med < 0.010x and max < 0.050x, depth med < 0.005x (q5_K: 0.15),
depth_conf med < 0.050x; x = quant scale (f32/f16 1, q8_0 3, q5_K 6).
Per-cell numbers: results/vggt-omega/gate_matrix_log.txt and
charts/vggt-omega/gate_summary.json (rendered into pose_error_heatmap /
quant_pareto_3d).

Findings of the 2026-09-29/30 re-verification:
- Bit-identical A/B: a worktree build at 36e621b (pre "opt vggt")
  reproduces the current outputs exactly — the 09-25 commits (opt vggt,
  C API) changed no numerics; the convT k==s fast path was already part of
  the 09-19 era binaries (see RESULTS.md §1).
- The q5_K depth floor is FRAME-dependent: ~0.0027 med_rel on benign random
  frames (seed 7) vs ~0.10 on the LCG parity frames (heavy tail of
  rel = |d-r|/max(|r|,0.05) on OOD inputs). Real-data impact is far
  smaller: on the official ETH3D 130 sets the q5_K AbsRel regression vs
  torch stays <= 9% relative on every metric — q6_K is the ~1 GB sweet
  spot (ETH3D depth AbsRel matches torch f32 exactly, ATE beats it); then
  q8_0/f16. q5_K was removed for omega the same day (dominated by q6_K);
  pi3x/dust3r/mapanything followed. Final rule (2026-10-01): ONE measured
  K tier per model — pi3 and vggt-1b instead KEPT q5_K (pi3's is its best
  depth/rot/ATE/pointmaps tier, vggt-1b's has the best AUC5) and their
  q6_Ks were removed (no best metric / weakest tier on 5 of 7).
- Fresh 13-set ETH3D evals are byte-identical to the committed ones for all
  12 configs (512/416/text x torch/f16/q8_0/q5_K); the refreshed 130-set
  two-sided table matches the committed one to 4-5 decimals
  (AUC5 79.23/79.08, ATE 0.00655/0.00662).
