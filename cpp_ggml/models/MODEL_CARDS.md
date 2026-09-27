# Github: https://github.com/Asher-1/map-anything-ggml

# MODEL CARDS — all six supported models (ckpts, GGUF, accuracy, speed)

> This directory (`cpp_ggml/models/`) holds the official torch checkpoints
> (`pytorch/<model>/`, conversion-only) and all 24 GGUF files (`gguf/`, the
> C++ runtime input: 6 architectures x f32/f16/q8_0/q5_K). This document
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
| **vggt-omega** (default) | vggt (N-view) | **0.0302** | **0.0208** | **0.0065** | **0.56** | 0.762 | CUDA 0.87x | 4q x 3b all PASS (+ text cos 1.0) |
| pi3x | pi3 + metric scale | 0.0533 | 0.0369 | 0.0162 | 2.11 | 0.251 | CUDA 1.44-1.50x | 12/12 |
| mapanything | unified (N-view, AAT) | 0.0553 | 0.0469 | 0.0124 | 0.92 | **0.162** | CUDA 1.64-2.23x | 9/9 |
| pi3 | pi3 (N-view) | 0.0650 | 0.0485 | 0.0172 | 1.82 | 0.733 | CUDA ~1.0x | 12/12 |
| vggt-1b | vggt (N-view) | 0.0674 | 0.0541 | 0.0208 | 2.28 | 0.788 | CUDA 1.60-1.69x | 12/12 |
| dust3r | pair-wise ancestor | 0.1008 | 0.1033 | 0.0509 | 2.77 | 0.897 | **CUDA 1.71-1.75x** | 12/12 |

How to read the figures embedded in every card below:

- `e2e_latency_bar` — steady-state P50 latency (log axis): official torch
  fp32 vs cpp f32/f16/q8_0/q5_K, per backend;
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
| 3-4 GB VRAM | **512-q5_K** | 859 MB, 90 ms, visible but passing accuracy ladder |
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
- **GGUF**: f32 4.36 GB / f16 2.18 GB / q8_0 1.26 GB / q5_K 859 MB
- **Official protocol accuracy** (ETH3D, 130 sets, official dataset + 8 metrics):

| Metric | PyTorch f32 | f16 | q8_0 | q5_K | Official ref* |
|---|---|---|---|---|---|
| Pose AUC@5 up (x100) | 79.23 | **79.08** | 77.08 | 71.69 | 79.53 |
| Pose ATE RMSE down | 0.00655 | 0.00662 | 0.00648 | 0.00712 | 0.00999 |
| Depth AbsRel down | 0.0208 | 0.0209 | 0.0210 | 0.0221 | 0.0204 |
| Point AbsRel down | 0.0302 | 0.0302 | 0.0305 | 0.0328 | 0.0263 |

\* Official reproduction.md (retrained ckpt, 2-100 views); slightly wider
protocol, order-of-magnitude reference only.

- **Speed** (512x512x2, steady-state pure inference, RTX 4090):

| Format | PyTorch f32 | f16 | q8_0 | q5_K |
|---|---|---|---|---|
| CUDA | 80.7ms | 93.1ms | **87.0ms** | 89.9ms |
| Vulkan | — | 128.8ms | 130.3ms | 134.6ms |
| CPU | — | 17.8s | **12.3s** | 19.6s |

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
- **GGUF**: f32 4.36 GB / f16 2.18 GB / q8_0 1.26 GB / q5_K 859 MB
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
- **GGUF**: f32 5.15 GB / f16 2.58 GB / q8_0 1.47 GB / q5_K 995 MB (the
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
- **Gate**: 12/12 PASS (f16 pose max 0.0006-0.0025, depth med_rel
  0.06-0.29%; [`benchmarks/results/gate_matrix.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/gate_matrix.md))
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
- **Gate**: 12/12 PASS (f16 local_points med_rel 0.0004-0.0037)
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
- **GGUF**: f32 3.79 GB / f16 1.94 GB / q8_0 1.08 GB / q5_K 731 MB
- **Gate**: 12/12 PASS (thresholds calibrated to pi3x's own f16-KV noise
  floor; q5_K's ConvHead chain is sensitive to 4.5-bit weights —
  **q8_0 is the recommended quant**)
- **Official protocol** (130 sets, 518x336, two-sided):

| Metric | torch f32 | cpp f16 |
|---|---|---|
| pointmaps_abs_rel | 0.053340 | 0.053448 |
| z_depth_abs_rel | 0.036922 | 0.036927 |
| pose_ate_rmse | 0.016232 | 0.016280 |
| pose_auc_5 (x100) | 66.92 | **67.08** |
| rot_err_deg | 2.108 | 2.113 |
| metric_scale_abs_rel | 0.250733 | 0.251058 |

- **Paper protocol** (13 scenes): torch/cpp deltas <= 0.0021 on every
  metric
- **Real-scene reconstruction** (courtyard, window [5,0], 518x336):

| metric | torch | f16 | q8_0 | q5_K |
|---|---|---|---|---|
| AbsRel | 0.015779 | 0.015774 | 0.016317 | 0.017492 |
| d1 | 0.999368 | 0.999368 | 0.999368 | 0.999368 |
| chamfer | 0.011814 | 0.011735 | 0.011897 | 0.012413 |
| fscore | 0.981632 | 0.981998 | 0.981998 | 0.982204 |
| pose rot diff (deg) | — | 0.028 | 0.088 | 0.905 |
| pose trans diff (m) | — | 0.005 | 0.032 | 0.053 |

- **Speed** (518x518x2, P50, RTX 4090 — faster than official torch across
  all three backends; re-measured 2026-09-24, exclusive):

| Backend | torch fp32 | torch TF32 | f32 | f16 | q8_0 | q5_K |
|---|---|---|---|---|---|---|
| CUDA | 290.3ms | 242.2 | 265.4 | **201.2 (1.44x)** | **193.7 (1.50x)** | 196.0 (1.48x) |
| Vulkan | — | — | 248.4 | **239.0 (1.21x)** | 240.5 | 245.0 |
| CPU | 17526ms | — | 13986 (1.25x) | **12959 (1.35x)** | 14891 (1.18x) | 15187 (1.15x) |

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
- **GGUF**: f32 4.44 GB / f16 2.81 GB / q8_0 1.26 GB / q5_K 859 MB
- **Gate**: 9/9 PASS (f16/q8_0/q5_K x CPU/CUDA/Vulkan, f16 rays med_rel
  ~0.001-0.003; the f32 GGUF exists for the latency baseline column)
- **Official protocol** (MapAnything paper protocol = its paper protocol,
  130 sets, two-sided): deltas <= 0.3% on every metric (pointmaps
  0.05334/0.05345, AUC@5 66.9/67.1, metric_scale 0.2507/0.2511) —
  [`benchmarks/results/mapanything/bench_official_eth3d_mapanything.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/mapanything/bench_official_eth3d_mapanything.md)
- **Real-scene reconstruction** (courtyard): AbsRel 0.07717/0.07729/0.07804
  (f16/q8_0/q5_K) vs torch 0.07711 (delta < 0.13%); pose rot diff
  0.01-0.10°
- **Speed** (518x518x2, P50, RTX 4090; re-measured 2026-09-24, exclusive):

| Backend | torch fp32 | f32 | f16 | q8_0 | q5_K |
|---|---|---|---|---|---|
| CUDA | 244.0ms | 180.7 (1.35x) | **148.6 (1.64x)** | **115.6 (2.11x)** | **109.2 (2.23x)** |
| Vulkan | — | 151.2 (1.61x) | **137.7 (1.77x)** | 141.0 (1.73x) | 145.7 (1.67x) |
| CPU | 12814ms | 11515 (1.11x) | 11047 (1.16x) | **10813 (1.19x)** | 11642 (1.10x) |

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
- **GGUF**: f32 2.18 GB / f16 1.17 GB / q8_0 699 MB / q5_K 510 MB
- **Gate** (fixed 512x512 S=2 frames, torch f32 ref): 12/12 PASS;
  lp med_rel 0.0002 (f16/f32) -> 0.0016 (q8_0) -> 0.004-0.009 (q5_K);
  the most quant-robust architecture in the repo
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

- **Real-scene reconstruction** (courtyard, window [5,0], 512x336): cpp f16
  vs torch AbsRel 0.132052 vs 0.132059 (delta 0.005%); pose delta rot
  0.000° / trans 0.00003 m. q8_0/q5_K pose deltas 0.020° / 0.014°.
- **Speed** (512x512x2, RTX 4090; CPU re-measured exclusively 2026-09-24;
  [`benchmarks/results/dust3r/speed_dust3r.md`](https://github.com/Asher-1/map-anything-ggml/blob/main/cpp_ggml/benchmarks/results/dust3r/speed_dust3r.md)):

| Backend | torch fp32 | f32 | f16 | q8_0 | q5_K |
|---|---|---|---|---|---|
| CUDA | 148.2ms | 184.2 | **86.6 (1.71x)** | **84.9 (1.75x)** | 86.2 (1.72x) |
| Vulkan | — | 212.3 | 201.6 (0.74x) | 216.2 (0.69x) | 210.1 (0.71x) |
| CPU | 5559ms | 5233 (1.06x) | **5068 (1.10x)** | 5732 (0.97x) | 5653 (0.98x) |

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
| f16 | 1x | **accuracy-first default** | pose 0.0010 / depth 0.11% (gate) |
| q8_0 | 0.57x | near-f16 accuracy, **fastest** | pose 0.0022 / depth 0.33% |
| q5_K | 0.39x | VRAM constrained | pose 0.0177 / depth 0.37% |

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
