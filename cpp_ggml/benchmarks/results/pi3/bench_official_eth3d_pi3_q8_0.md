# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=pi3, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.733509 |
| pointmaps_abs_rel | 0.069590 |
| pointmaps_inlier_thres_103 | 0.659220 |
| z_depth_abs_rel | 0.050457 |
| z_depth_inlier_thres_103 | 0.511436 |
| ray_dirs_err_deg | 1.948680 |
| pose_ate_rmse | 0.018389 |
| pose_auc_5 | 52.923077 |
| rot_err_deg | 1.341748 |
| center_err | 5.714751 |
| rot_auc_30 | 84.641026 |
