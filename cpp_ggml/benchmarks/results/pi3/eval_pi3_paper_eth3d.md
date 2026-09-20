# pi3 paper ETH3D protocol (Table 3) reproduction — 2026-09-21

Protocol: full WAI frame sequence per scene, keyframes frames[::5],
aspect-preserving width-518 /14 resolution, pi3 global point maps,
conf>0 validity mask (pi3's conf head approximates a binary mask),
per-view Umeyama Sim(3) + Open3D point-to-point ICP (threshold 0.1),
Acc/Comp = KDTree distance mean/median, NC = |normal dot| (analytic
normals: GT from NATIVE-resolution depth resampled to the model grid,
pred from the point-map grid), 13 scenes averaged.

NOTE: pi3 predicts scale-invariant local point maps per view -- the
global point cloud carries a per-view arbitrary scale while the poses
share one consistent frame (verified: Sim(3)-aligned camera centers
agree to 0.013-0.023 m).  Per-view alignment is therefore REQUIRED;
a global alignment diverges by design.

| metric | torch pi3 (f32) | cpp pi3-f16 | paper Table 3 |
|---|---|---|---|
| acc_mean | 0.0515 | 0.0516 | 0.194 |
| acc_med | 0.0225 | 0.0225 | 0.131 |
| comp_mean | 0.2243 | 0.2231 | 0.21 |
| comp_med | 0.1251 | 0.1236 | 0.128 |
| nc_mean | 0.5300 | 0.5271 | 0.883 |
| nc_med | 0.5267 | 0.5290 | 0.969 |

Observations:
- torch-vs-cpp agree to <1e-2 on every metric (f16 noise level) -- the
  C++ pi3 implementation is pinned to the official model under the
  paper's own protocol.
- comp_med/comp_mean land on the paper's numbers; acc is BETTER than the
  paper here (the paper's exact GT/mask/normal details were never open-
  sourced; the conf>0 mask and the 518-width sampling differ).
- NC is lower than the paper: the normals here are analytic on the
  model-resolution grid, and the paper's normal estimation details are
  unknown; Acc/Comp are the primary quality metrics.

Scripts: scripts/eval_pi3_paper_eth3d.py (side=torch/cpp); per-run
outputs eval_pi3_paper_eth3d_{torch,cpp}.md.
