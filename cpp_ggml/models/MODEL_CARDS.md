# MODEL CARDS — VGGT-Omega GGUF model overview

> This directory (`cpp_ggml/models/`) holds the official checkpoints
> (`pytorch/`, conversion-only) and all 12 GGUF files (`gguf/`, used by the
> C++ runtime). This document covers per-model use cases, size/VRAM, official
> protocol accuracy and measured speed, with references to the measured
> charts in `benchmarks/charts/`.

## 1. Quick selection

| Your scenario | Pick | Why |
|---|---|---|
| >= 6 GB VRAM, accuracy first | **512-f16** | flagship resolution, closest to PyTorch on every metric |
| 4-6 GB VRAM / best value | **512-q8_0** | half the size, fastest (87 ms), near-f16 accuracy |
| 3-4 GB VRAM | **512-q5_K** | 858 MB, 90 ms, visible but passing accuracy ladder |
| Reproduce the paper's 416-reproduce numbers | **416-reproduce-f16** | official reproduction ckpt |
| Low-res inputs / text-alignment research | **256-text-f16** | 256-resolution ckpt (C++ builds the TextAlignmentHead branch and emits `.text_embedding.bin`) |
| Two-view pointmaps / family-ancestor research | **dust3r-f16** | pair-wise (S=2), no pose head, best quant robustness (q8_0 ~0.2%) |
| No GPU | any + `--backend cpu` | q8_0 fastest (12.3 s) |

## 2. The model cards

### 1. vggt-omega-1b-512 (flagship, default)

- **ckpt**: `vggt_omega_1b_512.pt` (official facebook/VGGT-Omega release)
- **Default inference resolution**: 512 (balanced mode adapts the token
  budget to each input's aspect ratio)
- **Use cases**: general scene reconstruction (indoor/outdoor, 2-100 views),
  depth estimation, camera pose estimation
- **GGUF**: f32 4.36 GB / f16 2.18 GB / q8_0 1.25 GB / q5_K 858 MB
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
| CUDA | 80.7ms | 93.0ms | **87.0ms** | 89.9ms |
| Vulkan | — | 128.8ms | 130.3ms | 134.6ms |
| CPU | — | 17.8s | **12.3s** | 19.6s |

![latency](../benchmarks/charts/e2e_latency_bar.png)

### 2. vggt-omega-1b-416-reproduce (paper reproduction)

- **ckpt**: `vggt_omega_1b_416_reproduce.pt`
- **Default resolution**: 416
- **Use case**: reproducing the official paper/tech-report evaluation numbers
  (this is the ckpt behind reproduction.md)
- **GGUF**: one each of f32 / f16 / q8_0 / q5_K
- **Official protocol**: see `benchmarks/results/eval_eth3d_416_*.md` and
  `benchmarks/results/eval_eth3d_matrix.md`; accuracy families match PyTorch.

### 3. vggt-omega-1b-256-text (low-res + alignment head)

- **ckpt**: `vggt_omega_1b_256_text.pt`
- **Default resolution**: 256
- **Use cases**: low-resolution inputs, VRAM-constrained devices,
  text-alignment research
- **Note**: since 2026-09-24 the C++ graph implements the official
  TextAlignmentHead (enabled by the GGUF `vggt.enable_text_alignment`
  flag): the language-aligned embedding is emitted as
  `.text_embedding.bin` (2048-dim, L2-normalized). Gate: 256-text f16
  PASS with **text cosine = 1.000000** vs the official torch head
  (see `benchmarks/results/gate_matrix.md`)
- **GGUF**: one each of f32 / f16 / q8_0 / q5_K
- **Official protocol**: `benchmarks/results/eval_eth3d_text_*.md` (the
  text_torch column previously had a wrong protocol; all columns now use the
  real 256-text weights)

### 4. dust3r-512_dpt (pair-wise ancestor, M5)

- **ckpt**: `dust3r/model.safetensors` + `config.json` (official
  naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt; HF PyTorchModelHubMixin layout,
  no .pth in the repo)
- **Default resolution**: 512 (patch 16; pair-wise — exactly S=2 inputs)
- **Use cases**: two-view pointmap regression, the family's common ancestor;
  both pointmaps are emitted in view1's camera frame (s0 = head1
  self-view, s1 = head2 other-view); conf = 1+exp; **no pose** (the model
  has no pose head — official poses come from the out-of-network
  global-alignment optimizer, not ported; the CLI emits no .pose.bin)
- **GGUF**: f32 2.28 GB / f16 1.23 GB / q8_0 732 MB / q5_K 534 MB
- **Gate** (fixed 512x512 S=2 frames, torch f32 ref): 12/12 PASS;
  lp med_rel 0.0002 (f16/f32) -> 0.0016 (q8_0) -> 0.004-0.009 (q5_K);
  the most quant-robust architecture in the repo
  (`benchmarks/results/gate_matrix.md`, charts in `benchmarks/charts/dust3r/`)
- **Speed** (512x512x2, RTX 4090; `benchmarks/results/dust3r/speed_dust3r.md`):

| Format | PyTorch f32 | f16 | q8_0 | q5_K |
|---|---|---|---|---|
| CUDA | 148.2 ms | **86.6 ms (1.71x)** | 84.9 ms (1.75x) | 86.2 ms (1.72x) |
| CPU | 5558.9 ms | 4790.4 ms (1.16x) | 5648.8 ms (0.98x) | 5640.8 ms (0.99x) |

## 3. Quantization formats

| Format | Relative size | Positioning | Accuracy |
|---|---|---|---|
| f32 | 2x f16 | reference baseline (numeric oracle) | bit-locked to torch weights |
| f16 | 1x | **accuracy-first default** | pose 0.0010 / depth 0.11% (gate) |
| q8_0 | 0.57x | near-f16 accuracy, **fastest** | pose 0.0022 / depth 0.33% |
| q5_K | 0.39x | VRAM constrained | pose 0.0177 / depth 0.37% |

- q4_K was removed on 2026-09-18 (not a valid Pareto point), see
  `benchmarks/RESULTS.md` §6.
- Quantization applies only to 2-D Linear weights; LayerNorm/conv/RoPE stay
  floating point.

![quant pareto](../benchmarks/charts/quant_pareto_3d.png)

## 4. Dual-runtime agreement (illustrated)

**Real-data end-to-end reconstruction** (ETH3D courtyard, official protocol
window, four sources side by side):

![depth](../benchmarks/charts/recon_depth_comparison.png)

**Official-style world-frame colored point clouds** (GT / PyTorch / f16 /
q8_0 / q5_K):

![cloud](../benchmarks/charts/recon_pointcloud_comparison.png)

**Official 8-metric bars (x quantization x backend)**:

![metrics](../benchmarks/charts/recon_metrics_comparison.png)

**Per-op parity scatter** (random frames 512x512x2, C++ vs PyTorch):

![parity](../benchmarks/charts/parity_scatter.png)

**Official 8-metric global error heatmap**:

![pose heatmap](../benchmarks/charts/pose_error_heatmap.png)

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
