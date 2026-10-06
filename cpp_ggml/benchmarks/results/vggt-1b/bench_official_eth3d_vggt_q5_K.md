# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=vggt, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.788555 |
| pointmaps_abs_rel | 0.074521 |
| pointmaps_inlier_thres_103 | 0.621629 |
| z_depth_abs_rel | 0.052132 |
| z_depth_inlier_thres_103 | 0.532150 |
| ray_dirs_err_deg | 0.918392 |
| pose_ate_rmse | 0.024029 |
| pose_auc_5 | 63.692308 |
| rot_err_deg | 2.523399 |
| center_err | 4.812389 |
| rot_auc_30 | 88.282051 |
