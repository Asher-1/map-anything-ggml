# pi3x inference latency — cpp_ggml backends x quant vs official torch (2026-09-22, rev2)

Setup: 518x518, S=2 views, fixed frames518.bin; steady-state P50 after
warmup (CLI --warmup/--repeats in-process for ggml CUDA-graph replay;
torch timed with cuda.synchronize). Official torch baseline runs fp32 at
its native precision (matmul TF32 disabled by default; a --tf32 run is
listed separately); cpp entries use the listed weight quantization.

REV2 notes (CPU rows corrected): the ggml library default CPU thread
count is GGML_DEFAULT_N_THREADS == 4, which left the first CPU
measurements ~2x off. The CLI now defaults to hardware_concurrency
(--threads overrides; numerics verified thread-invariant: gate cell
bit-identical at 4 vs 32 threads).

## Latency (ms, P50) — speedup vs official torch CUDA fp32 (290.3 ms)

| backend | torch fp32 | torch fp32+TF32 | cpp f16 | cpp q8_0 | cpp q5_K |
|---|---|---|---|---|---|
| CUDA    | 290.3 | 242.2 | **201.2 (1.44x)** | **193.7 (1.50x)** | **196.0 (1.48x)** |
| Vulkan  | — | — | **239.0 (1.21x)** | 240.5 | 245.0 |
| CPU     | 17525.5 | — | **12047.0 (1.46x)** | **12808.5 (1.37x)** | **13156.6 (1.33x)** |

## Key facts

- CUDA: cpp is 1.44-1.50x faster than the official fp32 torch forward
  and still 1.19-1.25x faster than the TF32 torch path (f16/q8_0 weight
  streaming + CUDA graph replay).
- Vulkan: 1.21x faster than official torch CUDA fp32; within 1.2x of
  the CUDA path (no torch counterpart exists for Vulkan).
- CPU: with the corrected thread default cpp BEATS torch oneDNN fp32 by
  1.33-1.46x (ggml q8_0/f16 GEMM thread scaling is near-linear to 24+
  threads). Old measurements (24.9-33.4 s) were the 4-thread default.
- Recommendation: q8_0 (CUDA best latency; q5_K's ConvHead chain is
  4.5-bit sensitive on quality, see gate_matrix.md).

Raw JSONs: benchmarks/results/latency_pi3x_518x518x2_{CUDA0,Vulkan0,CPU}.json
(CPU json is the rev1 4-thread run — superseded by the table above),
bench_torch_pi3x_{cuda,cpu,tf32→cuda_tf32}.json. Harnesses:
benchmarks/bench_latency.py + scripts/bench_torch_pi3x.py (--tf32 flag).

> REV3 (2026-09-24): the CPU rows below were re-measured exclusively
(an earlier batch ran concurrently with a full-core build and was
contaminated); the authoritative table now lives in the per-model
report under benchmarks/charts/<model>/<model>_report.md, which also
adds the cpp f32 column and (dust3r) the Vulkan row.
