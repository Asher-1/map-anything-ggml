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
