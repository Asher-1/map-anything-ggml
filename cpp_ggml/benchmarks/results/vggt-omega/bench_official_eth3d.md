# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=pi3x, torch f32 vs cpp gguf)

sets evaluated: 10 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | torch | cpp |
|--------|--------|--------|
| metric_scale_abs_rel | 0.267911 | 0.268993 |
| pointmaps_abs_rel | 0.033110 | 0.033099 |
| pointmaps_inlier_thres_103 | 0.768716 | 0.768268 |
| z_depth_abs_rel | 0.030539 | 0.030449 |
| z_depth_inlier_thres_103 | 0.670988 | 0.678474 |
| ray_dirs_err_deg | 0.590627 | 0.589763 |
| pose_ate_rmse | 0.009796 | 0.009505 |
| pose_auc_5 | 64.000000 | 64.000000 |
| rot_err_deg | 0.738685 | 0.922957 |
| center_err | 5.541435 | 5.429519 |
| rot_auc_30 | 82.666667 | 83.333333 |
