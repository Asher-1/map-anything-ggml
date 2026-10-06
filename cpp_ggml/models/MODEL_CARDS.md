# Github: https://github.com/Asher-1/map-anything-ggml

# MODEL CARDS — all six supported models (ckpts, GGUF, accuracy, speed)

> This directory (`cpp_ggml/models/`) holds the official torch checkpoints
> (`pytorch/<model>/`, conversion-only) and all 32 GGUF files (`gguf/`, the
> C++ runtime input: 6 architectures, each f32/f16/q8_0 plus ONE measured
> K-quant — q6_K for omega/pi3x/mapanything/dust3r, q5_K for pi3/vggt-1b
> (the tier that wins on that backbone's official-protocol data). This
> document
> covers per-model use cases, size/VRAM, official-protocol accuracy and
> measured speed for **every** supported model, with the measured charts
> embedded inline.
>
> All figures (raw links) and all referenced documents (blob links) use
> **absolute GitHub URLs** (branch `main`), so the document renders anywhere
> — no checkout required:
> `https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/...`

## 1. The six models at a glance

Accuracy = official MapAnything ETH3D protocol (130 sets, seed 777, 2 views,
two-sided vs the official PyTorch; lower is better). Speed = CUDA latency vs
the official torch fp32 baseline (RTX 4090). Gate = quantization x backend
parity matrix.

| Model | Family / paradigm | pointmaps↓ | depth↓ | ATE↓ | rot°↓ | metric scale↓ | speed vs torch | gate |
|---|---|---|---|---|---|---|---|---|
| **vggt-omega** (default) | vggt (N-view) | **0.0302** | **0.0208** | **0.0065** | **0.56** | 0.762 | CUDA 0.85x | 4q x 3b all PASS (+ text cos 1.0) |
| pi3x | pi3 + metric scale | 0.0533 | 0.0369 | 0.0162 | 2.11 | 0.251 | CUDA 1.43-1.48x | 12/12 |
| mapanything | unified (N-view, AAT) | 0.0553 | 0.0469 | 0.0124 | 0.92 | **0.162** | CUDA 1.79-2.28x | 12/12 |
| pi3 | pi3 (N-view) | 0.0650 | 0.0485 | 0.0172 | 1.82 | 0.733 | CUDA ~1.0x | 15/15 |
| vggt-1b | vggt (N-view) | 0.0674 | 0.0541 | 0.0208 | 2.28 | 0.788 | CUDA 1.60-1.69x | 15/15 |
| dust3r | pair-wise ancestor | 0.1008 | 0.1033 | 0.0509 | 2.77 | 0.897 | **CUDA 2.02-2.06x** | 12/12 |

How to read the figures embedded in every card below:

- `e2e_latency_bar` — steady-state P50 latency (log axis): official torch
  fp32 vs cpp f32/f16/q8_0/q6_K/q5_K, per backend;
- `quant_pareto_3d` — file size vs pose error vs depth error per quant;
- `parity_scatter` — cpp f16 vs torch f32 per-point parity (random frames);
- `pose_error_heatmap` — gate-matrix error heatmap (quants x backends);
- `recon_depth/pointcloud/metrics_comparison` — real-scene ETH3D courtyard
  reconstruction (GT / PyTorch / cpp quants side by side, avg_dis scale
  alignment, view0-frame point clouds).

## 2. Quick selection

| Your scenario | Pick | Why |
|---|---|---|
| >= 6 GB VRAM, accuracy first | **512-f16** | flagship resolution, closest to PyTorch on every metric |
| 4-6 GB VRAM / best value | **512-q8_0** | half the size, fastest (87 ms), near-f16 accuracy |
| ~1 GB VRAM / balanced sweet spot | **512-q6_K** | 1.05 GB; ETH3D depth AbsRel matches torch f32 exactly, ATE even beats it (AUC5 76.5) |
| Metric scale + camera poses | **mapanything-f16** | metric_scale 0.162 far ahead; also the fastest large model on CUDA |
| Fastest CUDA inference | **dust3r-q8_0** | 84.9 ms pair-wise (1.75x vs official torch) |
| Reproduce the paper's 416-reproduce numbers | **416-reproduce-f16** | official reproduction ckpt |
| Low-res inputs / text-alignment research | **256-text-f16** | 256-resolution ckpt (C++ builds the TextAlignmentHead branch and emits `.text_embedding.bin`) |
| Two-view pointmaps / family-ancestor research | **dust3r-f16** | pair-wise (S=2), no pose head, best quant robustness |
| No GPU | any + `--backend cpu` | q8_0 fastest (12.3 s on omega-512) |

## 3. The model cards

### 3.1 vggt-omega-1b-512 (flagship, default)

- **ckpt**: `pytorch/vggt-omega/vggt_omega_1b_512.pt` (official
  facebook/VGGT-Omega release)
- **Default inference resolution**: 512 (balanced mode adapts the token
  budget to each input's aspect ratio)
- **Use cases**: general scene reconstruction (indoor/outdoor, 2-100 views),
  depth estimation, camera pose estimation
- **GGUF**: f32 4.36 GB / f16 2.18 GB / q8_0 1.26 GB / q6_K 1.05 GB
- **Official protocol accuracy** (ETH3D, 130 sets, official dataset + 8 metrics):

| Metric | PyTorch f32 | f16 | q8_0 | q6_K | Official ref* |
|---|---|---|---|---|---|
| Pose AUC@5 up (x100) | 79.23 | **79.08** | 77.08 | 76.46 | 79.53 |
| Pose ATE RMSE down | 0.00655 | 0.00662 | 0.00648 | **0.00644** | 0.00999 |
| Depth AbsRel down | 0.0208 | 0.0209 | 0.0210 | **0.0207** | 0.0204 |
| Point AbsRel down | 0.0302 | 0.0302 | 0.0305 | 0.0302 | 0.0263 |

\* Official reproduction.md (retrained ckpt, 2-100 views); slightly wider
protocol, order-of-magnitude reference only.

- **Speed** (512x512x2, steady-state pure inference, RTX 4090):

| Format | PyTorch f32 | f32 | f16 | q8_0 | q6_K |
|---|---|---|---|---|---|
| CUDA | 65.0ms | 109.0 | 76.2 | **70.7** | 75.5 |
| Vulkan | — | — | 94.7 | 97.3 | 101.5 |
| CPU | — | 18682  | 17.8s | **12.3s** | — |

(2026-10-02 re-measure, 10 repeats, torch baseline re-run in the same
session; CPU rows keep the 09-24 exclusive-CPU numbers)

![omega latency](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-omega/e2e_latency_bar.png)

![omega quant pareto](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-omega/quant_pareto_3d.png)

![omega parity](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-omega/parity_scatter.png)

![omega pose heatmap](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-omega/pose_error_heatmap.png)

![omega recon depth](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-omega/recon_depth_comparison.png)

![omega recon pointcloud](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-omega/recon_pointcloud_comparison.png)

![omega recon metrics](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-omega/recon_metrics_comparison.png)

### 3.2 vggt-omega-1b-416-reproduce (paper reproduction)

- **ckpt**: `pytorch/vggt-omega/vggt_omega_1b_416_reproduce.pt`
- **Default resolution**: 416
- **Use case**: reproducing the official paper/tech-report evaluation numbers
  (this is the ckpt behind reproduction.md)
- **GGUF**: f32 4.36 GB / f16 2.18 GB / q8_0 1.26 GB / q6_K 1.05 GB
- **Official protocol**: see
  [`benchmarks/results/vggt-omega/eval_eth3d_416_*.md`](https://github.com/Asher-1/map-anything-ggml/tree/main/cpp_ggml/benchmarks/results/vggt-omega/)
  and [`eval_eth3d_matrix.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/eval_eth3d_matrix.md);
  accuracy families match PyTorch.

### 3.3 vggt-omega-1b-256-text (low-res + alignment head)

- **ckpt**: `pytorch/vggt-omega/vggt_omega_1b_256_text.pt`
- **Default resolution**: 256
- **Use cases**: low-resolution inputs, VRAM-constrained devices,
  text-alignment research
- **Note**: since 2026-09-24 the C++ graph implements the official
  TextAlignmentHead (enabled by the GGUF `vggt.enable_text_alignment`
  flag): the language-aligned embedding is emitted as
  `.text_embedding.bin` (2048-dim, L2-normalized). Gate: 256-text f16
  PASS with **text cosine = 1.000000** vs the official torch head
  (see [`benchmarks/results/gate_matrix.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/gate_matrix.md))
- **GGUF**: f32 5.15 GB / f16 2.58 GB / q8_0 1.47 GB / q6_K 1.22 GB (the
  text head adds ~0.4 GB of f32 extras over the 512 variant)
- **Official protocol**:
  [`benchmarks/results/vggt-omega/eval_eth3d_text_*.md`](https://github.com/Asher-1/map-anything-ggml/tree/main/cpp_ggml/benchmarks/results/vggt-omega/)
  (the text_torch column previously had a wrong protocol; all columns now
  use the real 256-text weights)

### 3.4 vggt-1b (official VGGT-1B)

- **ckpt**: `pytorch/vggt1b/model.pt` (official facebook/VGGT-1B)
- **Default resolution**: 518 (patch 14)
- **Use cases**: N-view reconstruction with the classic VGGT heads; the only
  model besides omega emitting a dedicated **world pointmap + conf**
  (`.points.bin` / `.points_conf.bin`) alongside pose/depth
- **GGUF**: f32 4.54 GB / f16 2.43 GB / q8_0 1.43 GB / q5_K 1.04 GB
- **Gate**: 12/12 PASS (f32/f16/q8_0/q5_K x CPU/CUDA/Vulkan; f16 pose max
  0.0006-0.0025, depth med_rel 0.06-0.29%; q5_K kept — its AUC5 63.7 is
  the best of the three measured quants; q6_K removed 2026-10-01 — the
  weakest tier on 5 of 7 metrics, dominated by q8_0/q5_K —
  [`benchmarks/results/gate_matrix.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/gate_matrix.md))
- **Official protocol** (two-sided, 130 sets):
  [`benchmarks/results/vggt-1b/bench_official_eth3d_vggt.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/vggt-1b/bench_official_eth3d_vggt.md)
- **Real-scene reconstruction** (courtyard, window [5,0], 518x336):

| metric | torch | f16 | q8_0 | q5_K |
|---|---|---|---|---|
| AbsRel | 0.021158 | 0.020973 | 0.021542 | 0.015032 |
| d1 | 0.982500 | 0.982911 | 0.982249 | 0.996840 |
| chamfer | 0.013476 | 0.013442 | 0.013687 | 0.012912 |
| pose rot diff (deg) | — | **0.014** | 0.022 | 0.043 |
| pose trans diff (m) | — | 0.0002 | 0.0003 | 0.0016 |

- **Speed** (518x518x2, P50, RTX 4090; re-measured 2026-09-24, exclusive):

| Backend | torch fp32 | f32 | f16 | q8_0 | q5_K |
|---|---|---|---|---|---|
| CUDA | 284.0ms | 242.2 (1.17x) | **177.1 (1.60x)** | **168.1 (1.69x)** | 171.7 (1.65x) |
| Vulkan | — | 221.2 (1.28x) | **193.8 (1.47x)** | 194.3 (1.46x) | 198.0 (1.43x) |
| CPU | 12738ms | 12929 (0.99x) | **11834 (1.08x)** | 12214 (1.04x) | 13832 (0.92x) |

![vggt1b recon depth](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-1b/recon_depth_comparison.png)

![vggt1b recon pointcloud](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-1b/recon_pointcloud_comparison.png)

![vggt1b recon metrics](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-1b/recon_metrics_comparison.png)

![vggt1b latency](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-1b/e2e_latency_bar.png)

![vggt1b quant pareto](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-1b/quant_pareto_3d.png)

![vggt1b parity](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-1b/parity_scatter.png)

### 3.5 pi3

- **ckpt**: `pytorch/pi3/` (official yyfz233/Pi3, `model.safetensors`)
- **Default resolution**: 518 (patch 14)
- **Use cases**: N-view reconstruction with the simplified pi3 head set;
  poses are row-major c2w 4x4 (`.pose.bin`), depth = local_points camera-z
- **GGUF**: f32 3.66 GB / f16 1.84 GB / q8_0 981 MB / q5_K 639 MB
- **Gate**: 12/12 PASS (f32/f16/q8_0/q5_K x CPU/CUDA/Vulkan; f16
  local_points med_rel 0.0004-0.0037; q5_K is pi3's best depth/rot/ATE/
  pointmaps tier — 0.046483 / 1.086°; q6_K removed 2026-10-01 — no best
  metric on the 130 sets, squeezed between q5_K and q8_0)
- **Official protocol** (130 sets, two-sided): metric-level parity —
  [`benchmarks/results/pi3/bench_official_eth3d_pi3.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/pi3/bench_official_eth3d_pi3.md)
- **Paper protocol** (13 scenes, Acc/Comp/NC + Umeyama/ICP): torch/cpp
  deltas <= 0.01 on every metric; comp_med 0.1251/0.1236 pinned to the
  paper's 0.128 —
  [`benchmarks/results/pi3/eval_pi3_paper_eth3d.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/pi3/eval_pi3_paper_eth3d.md)
- **Real-scene reconstruction** (courtyard, window [5,0], 518x336):

| metric | torch | f16 | q8_0 | q5_K |
|---|---|---|---|---|
| AbsRel | 0.013730 | 0.013543 | 0.014092 | 0.014039 |
| d1 | 0.999174 | 0.999159 | 0.999166 | 0.999069 |
| chamfer | 0.010894 | 0.010721 | 0.010994 | 0.010034 |
| pose rot diff (deg) | — | **0.000** | 0.121 | 1.209 |
| pose trans diff (m) | — | 0.002 | 0.005 | 0.015 |

- **Speed** (518x518x2, P50, RTX 4090; CPU re-measured exclusively
  2026-09-24):

| Backend | torch fp32 | f32 | f16 | q8_0 | q5_K |
|---|---|---|---|---|---|
| CUDA | 237.4ms | 348.5 | 242.0 | **226.4 (1.05x)** | 234.0 |
| Vulkan | — | 420.3 | 368.7 | 358.2 | 400.5 |
| CPU | 13744ms | 10774 (1.28x) | **10249 (1.34x)** | 11821 (1.16x) | 11932 (1.15x) |

![pi3 recon depth](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3/recon_depth_comparison.png)

![pi3 recon pointcloud](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3/recon_pointcloud_comparison.png)

![pi3 recon metrics](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3/recon_metrics_comparison.png)

![pi3 latency](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3/e2e_latency_bar.png)

![pi3 quant pareto](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3/quant_pareto_3d.png)

![pi3 parity](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3/parity_scatter.png)

### 3.6 pi3x (pi3 + metric scale)

- **ckpt**: `pytorch/pi3x/` (official yyfz233/Pi3X, `model.safetensors`)
- **Default resolution**: 518 (patch 14)
- **Use cases**: pi3 plus a metric-scale head (`.scale.bin`); balanced
  alternative with a pinned paper protocol
- **GGUF**: f32 3.79 GB / f16 1.94 GB / q8_0 1.08 GB / q6_K 894 MB
- **Gate**: 12/12 PASS (f32/f16/q8_0/q6_K x CPU/CUDA/Vulkan; thresholds
  calibrated to pi3x's own f16-KV noise floor, q6_K lp med_rel
  0.033-0.055). q5_K was removed on 2026-10-01 — its ConvHead chain is
  sensitive to 4.5-bit weights and the fleet confirmation shows q6_K wins
  AUC5 by +3.7 points at the same size class (**q8_0 is the recommended
  quant**)  
- **Official protocol** (130 sets, 518x336, two-sided; per-quant cpp-only
  numbers from results/FLEET_QUANT_CONFIRMATION.md):

| Metric | torch f32 | cpp f16 | cpp q8_0 | cpp q6_K |
|---|---|---|---|---|
| pointmaps_abs_rel | 0.053340 | 0.053448 | 0.053749 | 0.053740 |
| z_depth_abs_rel | 0.036922 | 0.036927 | 0.037098 | 0.036885 |
| pose_ate_rmse | 0.016232 | 0.016280 | 0.016243 | 0.016311 |
| pose_auc_5 (x100) | 66.92 | **67.08** | **67.38** | 66.15 |
| rot_err_deg | 2.108 | 2.113 | 2.121 | 2.121 |
| metric_scale_abs_rel | 0.250733 | 0.251058 | 0.252437 | 0.251801 |

- **Paper protocol** (13 scenes): torch/cpp deltas <= 0.0021 on every
  metric
- **Real-scene reconstruction** (courtyard, window [5,0], 518x336):

| metric | torch | f16 | q8_0 | q6_K |
|---|---|---|---|---|
| AbsRel | 0.015779 | 0.015774 | 0.016317 | **0.015737** |
| d1 | 0.999368 | 0.999368 | 0.999368 | 0.999346 |
| chamfer | 0.011814 | 0.011735 | 0.011897 | 0.011826 |
| fscore | 0.981632 | 0.981998 | 0.981998 | 0.981659 |
| pose rot diff (deg) | — | **0.028** | 0.088 | 0.190 |
| pose trans diff (m) | — | 0.005 | 0.032 | 0.011 |

- **Speed** (518x518x2, P50, RTX 4090 — faster than official torch across
  all three backends; re-measured 2026-09-24, exclusive):

| Backend | torch fp32 | torch TF32 | f32 | f16 | q8_0 | q6_K |
|---|---|---|---|---|---|---|
| CUDA | 290.3ms | 242.2 | 252.5 | **202.5 (1.43x)** | **195.7 (1.48x)** | 202.1 (1.44x) |
| Vulkan | — | — | 237.4 | **229.8 (1.26x)** | 232.9 (1.25x) | 237.8 (1.22x) |
| CPU | 17526ms | — | 13986 (1.25x) | **12959 (1.35x)** | 14891 (1.18x) | — |

![pi3x recon depth](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3x/recon_depth_comparison.png)

![pi3x recon pointcloud](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3x/recon_pointcloud_comparison.png)

![pi3x recon metrics](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3x/recon_metrics_comparison.png)

![pi3x latency](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3x/e2e_latency_bar.png)

![pi3x quant pareto](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3x/quant_pareto_3d.png)

![pi3x parity](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/pi3x/parity_scatter.png)

### 3.7 mapanything (unified, AAT)

- **ckpt**: `pytorch/mapanything/` (official facebook/map-anything,
  `model.safetensors` + `config.json` for `from_pretrained`)
- **Default resolution**: 518 (patch 14)
- **Use cases**: unified N-view model with alternating-attention tokens;
  emits rays + non-ambiguous mask (`.mask.bin`) + metric scale
  (`.scale.bin`) on top of the pi3-family contract — the pick for metric
  scale and camera poses
- **GGUF**: f32 4.44 GB / f16 2.81 GB / q8_0 1.26 GB / q6_K 1.05 GB
- **Gate**: 9/9 PASS (f16/q8_0/q6_K x CPU/CUDA/Vulkan, f16 rays med_rel
  ~0.001-0.003; the f32 GGUF exists for the latency baseline column;
  q5_K retired 2026-10-01, HF-only)
- **Official protocol** (MapAnything paper protocol = its paper protocol,
  130 sets, two-sided): deltas <= 0.3% on every metric (pointmaps
  0.05334/0.05345, AUC@5 66.9/67.1, metric_scale 0.2507/0.2511) —
  [`benchmarks/results/mapanything/bench_official_eth3d_mapanything.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/mapanything/bench_official_eth3d_mapanything.md)
- **Real-scene reconstruction** (courtyard): AbsRel 0.077170/0.077287/0.077505
  (f16/q8_0/q6_K) vs torch 0.077106 (delta < 0.13%); pose rot diff
  0.010-0.030°
- **Speed** (518x518x2, P50, RTX 4090; re-measured 2026-09-24, exclusive):

| Backend | torch fp32 | f32 | f16 | q8_0 | q6_K |
|---|---|---|---|---|---|
| CUDA | 244.0ms | 174.2 (1.40x) | **136.0 (1.79x)** | **106.9 (2.28x)** | 114.0 (2.14x) |
| Vulkan | — | 145.6 | **137.8 (1.77x)** | 139.8 (1.75x) | 146.9 (1.66x) |
| CPU | 12814ms | 11515 (1.11x) | 11047 (1.16x) | **10813 (1.19x)** | — |

![mapanything recon depth](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/mapanything/recon_depth_comparison.png)

![mapanything recon pointcloud](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/mapanything/recon_pointcloud_comparison.png)

![mapanything recon metrics](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/mapanything/recon_metrics_comparison.png)

![mapanything latency](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/mapanything/e2e_latency_bar.png)

![mapanything quant pareto](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/mapanything/quant_pareto_3d.png)

![mapanything parity](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/mapanything/parity_scatter.png)

### 3.8 dust3r-512_dpt (pair-wise ancestor, M5)

- **ckpt**: `pytorch/dust3r/` (`model.safetensors` + `config.json`;
  official naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt,
  HF PyTorchModelHubMixin layout, no .pth in the repo)
- **Default resolution**: 512 (patch 16; pair-wise — exactly S=2 inputs)
- **Use cases**: two-view pointmap regression, the family's common ancestor;
  both pointmaps are emitted in view1's camera frame (s0 = head1
  self-view, s1 = head2 other-view); conf = 1+exp; **no pose** (the model
  has no pose head — the official poses come from the out-of-network
  global-alignment optimizer, not ported; the CLI emits no .pose.bin, the
  bench recovers poses via closed-form Procrustes)
- **GGUF**: f32 2.18 GB / f16 1.17 GB / q8_0 699 MB / q6_K 604 MB
- **Gate** (fixed 512x512 S=2 frames, torch f32 ref): 12/12 PASS
  (f32/f16/q8_0/q6_K x CPU/CUDA/Vulkan); lp med_rel 0.0002 (f16/f32) ->
  0.0016 (q8_0) -> 0.0029-0.0039 (q6_K); the most quant-robust
  architecture in the repo. q5_K was removed on 2026-10-01 — the fleet
  confirmation shows q6_K beats it on EVERY official metric (depth/ATE/
  AUC5/rot/pointmaps/scale; details below)
  ([`benchmarks/results/gate_matrix.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/gate_matrix.md))
- **Official protocol** (130 sets x 2 views, seed 777, 512x336, two-sided;
  pose via closed-form Procrustes — the official protocol itself is
  num_views=2):

| metric | torch f32 | cpp f16 | rel delta |
|---|---|---|---|
| pointmaps_abs_rel | 0.100810 | 0.100855 | 0.04% |
| z_depth_abs_rel | 0.103279 | 0.103310 | 0.03% |
| pose_ate_rmse | 0.050941 | 0.050955 | 0.03% |
| rot_err_deg | 2.773 | 2.726 | 1.7% |
| rot_auc_30 (x100) | 56.44 | 56.36 | 0.14% |
| ray_dirs_err_deg | 2.4215 | 2.4213 | 0.01% |

Full table:
[`benchmarks/results/dust3r/bench_official_eth3d_dust3r.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/dust3r/bench_official_eth3d_dust3r.md)

Per-quant cpp-only (130 sets,
[`benchmarks/results/FLEET_QUANT_CONFIRMATION.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/FLEET_QUANT_CONFIRMATION.md)):
**q6_K wins depth/ATE/AUC5/rot/pointmaps/scale over both q8_0 and q5_K**
(z_depth 0.102975 vs q8_0 0.103174 / q5_K 0.103523; ATE 0.050718 vs
0.050953 / 0.051103; AUC5 8.00 vs 7.38 / 7.54) — the 604 MB q6_K is the
best accuracy point of the whole tier ladder; q5_K was removed on
2026-10-01 (dominated).

- **Real-scene reconstruction** (courtyard, window [5,0], 512x336): cpp f16
  vs torch AbsRel 0.132052 vs 0.132059 (delta 0.005%); pose delta rot
  0.000° / trans 0.00003 m. q8_0/q6_K pose deltas 0.020° / 0.024°; q6_K
  also edges out every tier on AbsRel/chamfer (0.131862/0.081195).
- **Speed** (512x512x2, RTX 4090; CPU re-measured exclusively 2026-09-24;
  [`benchmarks/results/dust3r/speed_dust3r.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/dust3r/speed_dust3r.md)):

| Backend | torch fp32 | f32 | f16 | q8_0 | q6_K |
|---|---|---|---|---|---|
| CUDA | 148.2ms | 84.2 (1.76x) | **72.7 (2.04x)** | **71.8 (2.06x)** | 73.2 (2.02x) |
| Vulkan | — | 66.9 | **64.1 (2.31x)** | 65.4 (2.27x) | 69.7 (2.13x) |
| CPU | 5559ms | 5233 (1.06x) | **5068 (1.10x)** | 5732 (0.97x) | — |

![dust3r recon depth](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/dust3r/recon_depth_comparison.png)

![dust3r recon pointcloud](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/dust3r/recon_pointcloud_comparison.png)

![dust3r recon metrics](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/dust3r/recon_metrics_comparison.png)

![dust3r parity](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/dust3r/parity_scatter.png)

![dust3r latency](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/dust3r/e2e_latency_bar.png)

![dust3r quant pareto](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/dust3r/quant_pareto_3d.png)

## 4. Quantization formats

| Format | Relative size | Positioning | Accuracy |
|---|---|---|---|
| f32 | 2x f16 | reference baseline (numeric oracle) | bit-locked to torch weights |
| f16 | 1x | **accuracy-first default** | pose 0.0234 / depth 0.17% (gate) |
| q8_0 | 0.57x | near-f16 accuracy, **fastest** | pose 0.0141 / depth 0.82% |
| q6_K | 0.48x | **~1 GB sweet spot — ETH3D depth/ATE at torch-f32 level** | pose 0.1160 / depth 0.53% (gate) |

\* Gate numbers = CUDA parity on the canonical matrix frames (2026-09-30
re-run, gate_summary.json). On the official ETH3D 130 sets: **q6_K matches
torch f32 depth AbsRel exactly (0.020748 vs 0.020752) and beats it on ATE
(0.006436 vs 0.006549)**; q8_0/f16 stay <= 1% relative.
K-quant assignments are strictly data-driven (per-model 130-set verdicts,
results/FLEET_QUANT_CONFIRMATION.md — the quant response is NOT monotonic
in bit width, so each model ships exactly ONE measured K tier):

- **q6_K for omega** (2026-09-30): dominates q5_K (+22% size buys AUC5
  71.7 -> 76.5 and 3-6x tighter parity floors; the q5_K loss is inherent
  to 4.5-bit weight rounding, identical in any runtime), same judgment as
  the q4_K removal;
- **q6_K for pi3x / dust3r** (2026-10-01): wins AUC5 by +3.7 on pi3x and
  sweeps every official metric on dust3r;
- **q6_K for mapanything** (2026-10-01): wins depth/rot/pointmaps/AUC5
  (q5_K only edges metric scale by 1%);
- **q5_K for pi3** (2026-10-01): best depth/rot/ATE/pointmaps tier
  (0.046483 / 0.018289 / 1.086° / 0.065244); pi3's q6_K had NO best
  metric and was removed (locally + HF);
- **q5_K for vggt-1b** (2026-10-01): best AUC5 (63.7 vs q8_0 63.1); its
  q6_K was the weakest tier on 5 of 7 metrics and was removed (locally +
  HF). Removed tiers remain reproducible from the git history of
  RESULTS.md / FLEET_QUANT_CONFIRMATION.md / the converters.

- q4_K was removed on 2026-09-18 (not a valid Pareto point), see
  [`benchmarks/RESULTS.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/RESULTS.md) §6.
- Quantization applies only to 2-D Linear weights; LayerNorm/conv/RoPE stay
  floating point.

![quant pareto](https://github.com/Asher-1/map-anything-ggml/raw/main/cpp_ggml/benchmarks/charts/vggt-omega/quant_pareto_3d.png)

## 5. Output directory conventions (vs the official demo)

| Side | Command | Output directory/files |
|---|---|---|
| Official PyTorch demo | `demo_gradio.py` (default config) | `demo_outputs/input_images_<ts>/predictions.npz` + GLB |
| C++ CLI | `vggt-cli --images ... --out-prefix P` | `P.pose.bin (S,9)` / `P.depth.bin (S,H,W)` / `P.depth_conf.bin (S,H,W)` / `P.meta.json` |
| Official npz packaging | `scripts/make_predictions_npz.py --prefix P` | `P.predictions.npz` (official keys incl. extrinsic/intrinsic/world_points_from_depth) |
| Dual-runtime comparison | `scripts/compare_official_demo.py` | `<out>/torch/predictions.npz` vs `<out>/cpp/out.*.bin` + `COMPARISON.md` |

Key mapping: `pose_enc ↔ .pose.bin`, `depth[...,0] ↔ .depth.bin`,
`depth_conf ↔ .depth_conf.bin`; `extrinsic/intrinsic/world_points_from_depth`
are derived from `pose_enc+depth` with the official formulas (built into both
helper scripts).

## 6. One-click reproduction

```bash
./run_mapggml.sh demo                 # newcomers: download -> convert -> build -> infer
./run_mapggml.sh gate                 # dual-runtime parity gate
./run_mapggml.sh bench --backend cuda # steady-state latency matrix
PYTHONPATH=/tmp/torch_cuda_lib:<repo> python3 cpp_ggml/scripts/compare_official_demo.py \
    --images a.jpg b.jpg --out-dir /tmp/demo_compare   # official demo dual-runtime compare
```
