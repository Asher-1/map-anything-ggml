# benchmarks — charts, tables and the cross-model comparison

Every model ships the full five-piece verification suite (see
[../FEATURE_PARITY_AUDIT.md](../FEATURE_PARITY_AUDIT.md)): the
quantization x backend gate matrix, the official-protocol ETH3D bench
(two-sided), the real-scene reconstruction comparison, latency benchmarks,
and an illustrated report. This README is the visual index.

## Cross-model comparison (official MapAnything ETH3D protocol, 130 sets, seed 777, 2 views)

Absolute accuracy on the official torch side (cpp<->torch two-sided deltas
are < 1% everywhere — the full two-sided tables are in `results/<model>/`):

| Model | pointmaps_abs_rel↓ | z_depth_abs_rel↓ | ATE RMSE↓ | rot_err°↓ | metric scale↓ | CUDA speedup |
|---|---|---|---|---|---|---|
| **vggt-omega** | **0.0302** | **0.0208** | **0.0065** | **0.56** | 0.762 | 0.87x |
| pi3x | 0.0533 | 0.0369 | 0.0162 | 2.11 | 0.251 | 1.44-1.50x |
| mapanything | 0.0553 | 0.0469 | 0.0124 | 0.92 | **0.162** | **1.64-2.23x** |
| pi3 | 0.0650 | 0.0485 | 0.0172 | 1.82 | 0.733 | ~1.0x |
| vggt-1b | 0.0674 | 0.0541 | 0.0208 | 2.28 | 0.788 | 1.60-1.69x |
| dust3r | 0.1008 | 0.1033 | 0.0509 | 2.77 | 0.897 | **1.71-1.75x** |

### Strengths, trade-offs and when to use which

| Model | Core strengths | Trade-offs | Best for |
|---|---|---|---|
| **vggt-omega** (default) | best accuracy on every metric; text-aligned embeddings (`.text_embedding.bin`); very quantization-robust | CUDA 0.87x vs its strong official baseline (the only model slower than official) | maximum-accuracy N-view reconstruction |
| **mapanything** | best metric scale (0.162) and near-best poses; fastest large model on CUDA (q8_0 2.11x); rays + non-ambiguous-mask outputs | geometry slightly behind pi3x | metric 3D, camera poses, AR/robotics deployments |
| **pi3x** | best geometry among the 130-set-native models; paper protocol pinned (<= 0.0021); CUDA 1.44-1.50x | scale head behind mapanything | balanced accuracy + speed without metric-scale needs |
| **pi3** | the minimal cross-view decoder; parity reference | slower than pi3x everywhere | lineage / ablation baseline |
| **vggt-1b** | official VGGT-1B; point-map head (`.points.bin`) | slowest N-view model | official-VGGT compatibility |
| **dust3r** | family ancestor; CUDA 1.71-1.75x; mildest quantization degradation; lowest CPU latency | pair-wise S=2 only; no pose head (Procrustes-recovered) | resource-constrained devices, throughput, two-view matching |

## Per-model charts (click through)

| Model | Latency (backends x quants) | Gate heatmap | Quant Pareto | Parity scatter | Recon |
|---|---|---|---|---|---|
| [vggt-omega](charts/vggt-omega/omega_report.md) | ![lat](charts/vggt-omega/e2e_latency_bar.png) | ![hm](charts/vggt-omega/pose_error_heatmap.png) | ![qp](charts/vggt-omega/quant_pareto_3d.png) | ![ps](charts/vggt-omega/parity_scatter.png) | ![rc](charts/vggt-omega/recon_depth_comparison.png) |
| [vggt-1b](charts/vggt-1b/vggt1b_report.md) | ![lat](charts/vggt-1b/e2e_latency_bar.png) | ![hm](charts/vggt-1b/pose_error_heatmap.png) | ![qp](charts/vggt-1b/quant_pareto_3d.png) | ![ps](charts/vggt-1b/parity_scatter.png) | ![rc](charts/vggt-1b/recon_depth_comparison.png) |
| [pi3](charts/pi3/pi3_report.md) | ![lat](charts/pi3/e2e_latency_bar.png) | ![hm](charts/pi3/pose_error_heatmap.png) | ![qp](charts/pi3/quant_pareto_3d.png) | ![ps](charts/pi3/parity_scatter.png) | ![rc](charts/pi3/recon_depth_comparison.png) |
| [pi3x](charts/pi3x/pi3x_report.md) | ![lat](charts/pi3x/e2e_latency_bar.png) | ![hm](charts/pi3x/pose_error_heatmap.png) | ![qp](charts/pi3x/quant_pareto_3d.png) | ![ps](charts/pi3x/parity_scatter.png) | ![rc](charts/pi3x/recon_depth_comparison.png) |
| [mapanything](charts/mapanything/mapanything_report.md) | ![lat](charts/mapanything/e2e_latency_bar.png) | ![hm](charts/mapanything/pose_error_heatmap.png) | ![qp](charts/mapanything/quant_pareto_3d.png) | ![ps](charts/mapanything/parity_scatter.png) | ![rc](charts/mapanything/recon_depth_comparison.png) |
| [dust3r](charts/dust3r/dust3r_report.md) | ![lat](charts/dust3r/e2e_latency_bar.png) | ![hm](charts/dust3r/pose_error_heatmap.png) | ![qp](charts/dust3r/quant_pareto_3d.png) | ![ps](charts/dust3r/parity_scatter.png) | ![rc](charts/dust3r/recon_depth_comparison.png) |

Each linked report contains: the gate matrix table, the official-protocol
two-sided metric table, the courtyard reconstruction figures
(`recon_{depth,pointcloud,metrics}_comparison.png` + `recon_comparison.md`)
and reproduction commands.

## Directory layout

```
results/<model>/   raw artifacts: gate logs, official-protocol benches,
                   latency JSONs (CPU/CUDA/Vulkan), torch baselines, speed notes
charts/<model>/    rendered figures + <model>_report.md + gate_summary.json
gate_matrix.md     the full (model x quant x backend) gate matrix
RESULTS.md         historical optimization records (vggt-omega journey)
```

Regenerate everything for one model:

```bash
python3 scripts/plot_charts_model.py --arch <arch> \
  --gate-log <gate.log> --torch-prefix <torch-dump> --cpp-prefix <cpp-dump>
PYTHONPATH=/tmp/torch_cuda_lib:. python3 scripts/compare_reconstruction_pi3x.py \
  --arch <arch> --data-root <eth3d> --metadata-dir <metadata>
```
