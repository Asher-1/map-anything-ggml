# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=mapanything, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | torch | cpp |
|--------|--------|--------|
| metric_scale_abs_rel | 0.161960 | 0.161861 |
| pointmaps_abs_rel | 0.055262 | 0.055244 |
| pointmaps_inlier_thres_103 | 0.617681 | 0.617827 |
| z_depth_abs_rel | 0.046921 | 0.046886 |
| z_depth_inlier_thres_103 | 0.532935 | 0.532974 |
| ray_dirs_err_deg | 1.108489 | 1.105751 |
| pose_ate_rmse | 0.012431 | 0.012415 |
| pose_auc_5 | 57.384615 | 58.000000 |
| rot_err_deg | 0.921459 | 0.888006 |
| center_err | 4.704435 | 4.688049 |
| rot_auc_30 | 86.846154 | 86.974359 |
