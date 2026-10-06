# benchmarks/results — per-model index

Matrix-level (cross-model) files live at the root of this directory; each
model's evaluation artifacts live in the same-named subdirectory.

## Root (cross-model matrices)

| File | Content |
|---|---|
| [gate_matrix.md](gate_matrix.md) | full (model x quant x backend) parity gate matrix, one measured K tier per model: pi3 12/12, pi3x 12/12, mapanything 9/9, vggt 12/12, dust3r 12/12 (q6_K for pi3x/mapanything/dust3r, q5_K for pi3/vggt-1b; pi3/vggt-1b q6_K removed 10-01 — dominated), vggt-omega 12/12 + 256-text PASS (text cos 1.0) |
| [../../FEATURE_PARITY_AUDIT.md](../../FEATURE_PARITY_AUDIT.md) | end-to-end feature-parity audit of every model vs the official repos (2026-09-24) |
| [eval_eth3d_matrix.md](eval_eth3d_matrix.md) | vggt-omega ETH3D quantization matrix (416/512/text variants under vggt-omega/) |

## [vggt-omega/](vggt-omega/)

- `gate_matrix_log.txt` — raw gate-matrix log (f32/f16/q8_0/q6_K x 3
  backends, 12/12 PASS; 2026-09-30 re-run)
- `bench_official_eth3d_130set.md` — official protocol, full 130 sets
  (filled in 2026-09-24; two-sided deltas < 1%, pose_auc_5 79.2 vs the
  official reference 79.53)
- `e2e_vggt-omega-1b-512_512x512x2.json` — end-to-end accuracy (vs official
  torch f32)
- `latency_vggt-omega-1b-512_*.json` — latency over 3 backends x quants x
  view counts
- `pytorch_baseline_512x512x2.json` — official torch baseline
- `eval_eth3d_{512,416}_*.md`, `eval_eth3d_text_*.md`, `eval_eth3d_torch.md`
  — per-table ETH3D quantization matrix
- `bench_official_eth3d{,_cpu,_q8_0,_q5_K,_vulkan}.md` — official-protocol
  torch/cpp tables per backend/quant
- `diagnose_depth_gap.md` — historical DP-head investigation
- Charts and illustrated report:
  [../charts/vggt-omega/omega_report.md](../charts/vggt-omega/omega_report.md);
  `eval_eth3d_torch.md` also records the T-position-encoding bug diagnosis

## [vggt-1b/](vggt-1b/)

- `gate_matrix_log.txt` — raw gate-matrix log (f32/f16/q8_0/q5_K x 3
  backends, 12/12 PASS; 2026-10-01 re-run with the shipped K tier)
- `bench_official_eth3d_vggt.md` — official-protocol two-sided bench for
  official VGGT-1B
- `latency_vggt-1b_518x518x2_*.json` + `bench_torch_vggt1b_cuda/cpu.json` —
  3-backend latency + official baseline
- Charts and illustrated report:
  [../charts/vggt-1b/vggt1b_report.md](../charts/vggt-1b/vggt1b_report.md)

## [pi3/](pi3/)

- `gate_matrix_log.txt` — raw gate-matrix log (f32/f16/q8_0/q5_K x 3
  backends, 12/12 PASS; 2026-10-01 re-run with the shipped K tier)
- `bench_official_eth3d_pi3.md` — official protocol, 130 sets, two-sided
  (metric-level parity)
- `eval_pi3_paper_eth3d{,_torch,_cpp}.md` — paper Table 3 protocol on 13
  scenes (comp pinned to the paper numbers)
- `latency_pi3_518x518x2_*.json` + `bench_torch_pi3_cuda/cpu.json` —
  3-backend latency + official baseline
- Charts and illustrated report:
  [../charts/pi3/pi3_report.md](../charts/pi3/pi3_report.md)

## [pi3x/](pi3x/)

- `gate_matrix_log.txt` — raw gate-matrix log (f32/f16/q8_0/q6_K x 3
  backends, 12/12 PASS; 2026-10-01 re-run with the shipped K tier)
- `bench_official_eth3d_pi3x.md` — official protocol, **full 130 sets**,
  two-sided
- `eval_pi3_paper_eth3d_pi3x_{torch,cpp}.md` — paper protocol on 13 scenes,
  two-sided (delta <= 0.0021)
- `eval_pi3_paper_eth3d_pi3_cpp.md` — pi3 re-confirmation after the reconversion
- `latency_pi3x_518x518x2_{CUDA0,Vulkan0,CPU}.json` — 3 backends x quants
- `bench_torch_pi3x_{cuda,cuda_tf32,cpu}.json` — official torch baseline
  (fp32 / TF32 / CPU)
- `speed_pi3x.md` — speed report (rev2: CPU rows corrected for the thread
  default)

Charts and illustrated report:
[../charts/pi3x/pi3x_report.md](../charts/pi3x/pi3x_report.md)

## [mapanything/](mapanything/)

- `gate_matrix_log.txt` — raw gate-matrix log (f16/q8_0/q6_K x 3 backends,
  9/9 PASS; 2026-10-01 re-run with the shipped K tier)
- `bench_official_eth3d_mapanything.md` — official protocol, 130 sets,
  two-sided (metric-level parity)
- `latency_mapanything_518x518x2_*.json` +
  `bench_torch_mapanything_{cuda,cpu}.json` — 3-backend latency + official
  baseline (CUDA q8_0 2.11x)
- Charts and illustrated report:
  [../charts/mapanything/mapanything_report.md](../charts/mapanything/mapanything_report.md)

## [dust3r/](dust3r/)

- `gate_matrix_log.txt` — raw gate-matrix log (f32/f16/q8_0 x 3 backends
  at recording time; q6_K floors in gate_matrix_q6k_fleet.log),
  12/12 PASS; q5_K removed 2026-10-01, dominated by q6_K on every
  official metric
- `bench_official_eth3d_dust3r.md` — official protocol, 130 sets, two-sided
  (num_views=2 is a natural fit for the pair-wise model; poses recovered by
  the closed-form rigid Procrustes over the two heads' pointmaps, shared
  implementation on both sides, no K / no GT)
- `latency_dust3r_512x512x2_*.json` + `bench_torch_dust3r_{cuda,cpu}.json` —
  2-backend latency + official baseline (CUDA f16 1.71x / q8_0 1.75x, CPU
  f16 1.16x)
- `speed_dust3r.md` — speed summary table and observations
- Charts and illustrated report:
  [../charts/dust3r/dust3r_report.md](../charts/dust3r/dust3r_report.md)
  (incl. the three courtyard recon figures + parity scatter)
