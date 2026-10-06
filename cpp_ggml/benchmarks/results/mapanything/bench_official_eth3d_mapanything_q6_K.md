# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=mapanything, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.164076 |
| pointmaps_abs_rel | 0.055542 |
| pointmaps_inlier_thres_103 | 0.618099 |
| z_depth_abs_rel | 0.047034 |
| z_depth_inlier_thres_103 | 0.529340 |
| ray_dirs_err_deg | 1.119588 |
| pose_ate_rmse | 0.012504 |
| pose_auc_5 | 56.307692 |
| rot_err_deg | 0.891136 |
| center_err | 4.744628 |
| rot_auc_30 | 86.692308 |
