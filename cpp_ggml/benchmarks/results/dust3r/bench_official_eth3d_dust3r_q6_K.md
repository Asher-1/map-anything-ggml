# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 512x336, arch=dust3r, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.897447 |
| pointmaps_abs_rel | 0.101118 |
| pointmaps_inlier_thres_103 | 0.516640 |
| z_depth_abs_rel | 0.102975 |
| z_depth_inlier_thres_103 | 0.302448 |
| ray_dirs_err_deg | 2.419849 |
| pose_ate_rmse | 0.050718 |
| pose_auc_5 | 8.000000 |
| rot_err_deg | 2.806158 |
| center_err | 16.632276 |
| rot_auc_30 | 56.435897 |
