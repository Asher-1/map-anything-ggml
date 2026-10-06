# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=pi3x, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.251801 |
| pointmaps_abs_rel | 0.053740 |
| pointmaps_inlier_thres_103 | 0.716421 |
| z_depth_abs_rel | 0.036885 |
| z_depth_inlier_thres_103 | 0.629006 |
| ray_dirs_err_deg | 0.845012 |
| pose_ate_rmse | 0.016311 |
| pose_auc_5 | 66.153846 |
| rot_err_deg | 2.120841 |
| center_err | 3.686129 |
| rot_auc_30 | 90.153846 |
