# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 512x336, arch=dust3r, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | torch | cpp |
|--------|--------|--------|
| metric_scale_abs_rel | 0.897498 | 0.897505 |
| pointmaps_abs_rel | 0.100810 | 0.100855 |
| pointmaps_inlier_thres_103 | 0.517461 | 0.517378 |
| z_depth_abs_rel | 0.103279 | 0.103310 |
| z_depth_inlier_thres_103 | 0.302864 | 0.302517 |
| ray_dirs_err_deg | 2.421495 | 2.421251 |
| pose_ate_rmse | 0.050941 | 0.050955 |
| pose_auc_5 | 7.692308 | 7.692308 |
| rot_err_deg | 2.772634 | 2.726468 |
| center_err | 16.625759 | 16.664271 |
| rot_auc_30 | 56.435897 | 56.358974 |
