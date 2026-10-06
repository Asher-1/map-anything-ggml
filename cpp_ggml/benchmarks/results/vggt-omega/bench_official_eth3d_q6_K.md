# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 512x336, arch=vggt_omega, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.761340 |
| pointmaps_abs_rel | 0.030212 |
| pointmaps_inlier_thres_103 | 0.857513 |
| z_depth_abs_rel | 0.020748 |
| z_depth_inlier_thres_103 | 0.798804 |
| ray_dirs_err_deg | 0.957834 |
| pose_ate_rmse | 0.006436 |
| pose_auc_5 | 76.461538 |
| rot_err_deg | 0.751499 |
| center_err | 1.801619 |
| rot_auc_30 | 94.948718 |
