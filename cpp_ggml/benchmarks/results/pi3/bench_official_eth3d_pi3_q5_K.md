# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=pi3, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.733816 |
| pointmaps_abs_rel | 0.065244 |
| pointmaps_inlier_thres_103 | 0.659022 |
| z_depth_abs_rel | 0.046483 |
| z_depth_inlier_thres_103 | 0.530639 |
| ray_dirs_err_deg | 1.742913 |
| pose_ate_rmse | 0.018289 |
| pose_auc_5 | 50.461538 |
| rot_err_deg | 1.086058 |
| center_err | 5.721955 |
| rot_auc_30 | 84.692308 |
