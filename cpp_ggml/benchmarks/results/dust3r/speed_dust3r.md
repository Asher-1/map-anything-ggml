# dust3r speed — cpp_ggml vs official PyTorch (2026-09-23)

Setup: DUSt3R_ViTLarge_BaseDecoder_512_dpt (571.2M params), 512x512 S=2
(pair-wise), steady-state pure inference (CLI --warmup 3 --repeats 10 on
CUDA / --warmup 2 --repeats 5 on CPU; torch mirrors with
torch.cuda.synchronize / fp32-faithful matmuls). RTX 4090.
Sources: latency_dust3r_512x512x2_{CPU,CUDA0}.json,
bench_torch_dust3r_{cuda,cpu}.json, chart e2e_latency_bar.png.

## CUDA (torch fp32-faithful = official behaviour)

| Format | PyTorch f32 | cpp f16 | cpp q8_0 | cpp q5_K |
|---|---|---|---|---|
| CUDA p50 | 148.2 ms | **86.6 ms (1.71x)** | **84.9 ms (1.75x)** | 86.2 ms (1.72x) |

## CPU

| Format | PyTorch f32 (oneDNN) | cpp f16 | cpp q8_0 | cpp q5_K |
|---|---|---|---|---|
| CPU p50 | 5558.9 ms | **4790.4 ms (1.16x)** | 5648.8 ms (0.98x) | 5640.8 ms (0.99x) |

## Observations

- CUDA: cpp is 1.71-1.75x faster than the official fp32 path — the largest
  CUDA win in the repo (pi3x 1.44-1.50x, mapanything 1.64-2.23x, vggt-omega
  0.87x). q8_0 is the fastest cell; f16 costs +2% and is the recommended
  default (parity indistinguishable from f32, see gate_matrix.md).
- CPU: f16 wins 1.16x; q8_0/q5_K are parity with torch oneDNN (the croco
  decoder's many small cross-attention matmuls are quant-dequant bound on
  CPU). CPU numbers use the default hardware_concurrency thread count.
- The torch CPU number reflects the slow-path RoPE2D (curope is CUDA-only);
  torch CUDA uses the same slow path yet stays fast on GPU.

> REV3 (2026-09-24): the CPU rows below were re-measured exclusively
(an earlier batch ran concurrently with a full-core build and was
contaminated); the authoritative table now lives in the per-model
report under benchmarks/charts/<model>/<model>_report.md, which also
adds the cpp f32 column and (dust3r) the Vulkan row.
