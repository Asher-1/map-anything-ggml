# vggt-omega evaluation report — C++ ggml vs official PyTorch

> Completed 2026-09-23 · official `facebook/VGGT-Omega` torch f32 (1B, the
> 512/416/text variants) vs cpp `vggt-omega-1b-512-{f16,q8_0,q5_K,f32}.gguf`
> · RTX 4090 · every number measured by this repo's own scripts (historical
> details in [RESULTS.md](../../RESULTS.md)).

## 1. Summary

| Dimension | Result |
|---|---|
| Gate (4 quants x CPU/CUDA/Vulkan, random 512x512x2 frames) | **all PASS** (f16 pose max 0.0007-0.0018, depth med 0.11-0.29%) |
| Official ETH3D protocol (two-sided + per-backend/quant tables) | metric-level parity (see the eval_eth3d_* / bench_official_eth3d_* series under results/vggt-omega/) |
| Real-scene reconstruction (courtyard) | see the recon figures below (same protocol as pi3/pi3x/vggt-1b/mapanything: avg_dis scale alignment) |
| Speed (512x512 S=2 P50) | CUDA f16 93.1ms vs official torch fp32 80.7ms (0.87x); Vulkan/CPU in the table below |

> Note: vggt-omega was the first model ported to this repo (optimization
> journey: 255.7ms -> 90ms, see RESULTS.md §2.1/§4.5 for the five steps);
> its official torch baseline at 512x512 S=2 is 80.7ms and the cpp steady
> state is slightly slower at 93ms (0.87x). The four models ported
> afterwards are all faster than their official baselines (1.05-2.23x),
> thanks to the unified merged-graph + CUDA graph replay infrastructure.

## 2. End-to-end accuracy charts

Gate passes on all three backends (pose < 0.005, depth_med < 0.005
thresholds; the quantization ladder is clearly visible):

| quant | pose max_abs | depth median_rel |
|---|---|---|
| f16   | 0.0010 | 0.11% |
| q8_0  | 0.0022 | 0.33% |
| q5_K  | 0.0177 | 0.37% |

![pose error heatmap](pose_error_heatmap.png)

![quant pareto](quant_pareto_3d.png)

![parity scatter](parity_scatter.png)

## 3. Official ETH3D protocol

Under results/vggt-omega/, split by resolution (512/416) and input form
(text): `eval_eth3d_512_*.md`, `eval_eth3d_416_*.md`,
`eval_eth3d_text_*.md`, `bench_official_eth3d{,_cpu,_q8_0,_q5_K,_vulkan}.md`.

## 4. Speed (P50, ms)

| Backend | torch fp32 (official) | cpp f32 | cpp f16 | cpp q8_0 | cpp q5_K |
|---|---|---|---|---|---|
| CUDA   | 80.7 | 132.3 | 93.1 | **87.0** | 89.9 |
| Vulkan | —    | —      | 128.8 | 130.3 | 134.6 |
| CPU    | —    | 18682  | 17781 | **12301** | 19558 |

![e2e latency](e2e_latency_bar.png)

Raw JSONs: results/vggt-omega/latency_vggt-omega-1b-512_512x512x2_*.json,
pytorch_baseline_512x512x2.json, e2e_vggt-omega-1b-512_512x512x2.json.
Throughput (S sweep): S=2 21.5 views/s -> S=8 17.3 views/s (RESULTS.md §5).

## 5. Real-scene end-to-end reconstruction comparison

![depth comparison](recon_depth_comparison.png)

![point cloud comparison](recon_pointcloud_comparison.png)

![reconstruction metrics](recon_metrics_comparison.png)

Per-view numbers: [recon_comparison.md](recon_comparison.md).

## 6. Reproduce

```bash
./run_mapggml.sh gate --models 512              # omega self-contained gate (generates the torch reference)
./run_mapggml.sh infer --models 512 --images a.jpg b.jpg
./run_mapggml.sh bench --models 512 --quants all
```
