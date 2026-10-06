# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=pi3x, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.252437 |
| pointmaps_abs_rel | 0.053749 |
| pointmaps_inlier_thres_103 | 0.713858 |
| z_depth_abs_rel | 0.037098 |
| z_depth_inlier_thres_103 | 0.626918 |
| ray_dirs_err_deg | 0.853116 |
| pose_ate_rmse | 0.016243 |
| pose_auc_5 | 67.384615 |
| rot_err_deg | 2.121493 |
| center_err | 3.684875 |
| rot_auc_30 | 90.358974 |
