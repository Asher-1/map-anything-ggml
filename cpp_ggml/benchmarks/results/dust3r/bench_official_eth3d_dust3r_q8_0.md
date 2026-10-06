# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 512x336, arch=dust3r, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.897440 |
| pointmaps_abs_rel | 0.100995 |
| pointmaps_inlier_thres_103 | 0.517138 |
| z_depth_abs_rel | 0.103174 |
| z_depth_inlier_thres_103 | 0.302254 |
| ray_dirs_err_deg | 2.422727 |
| pose_ate_rmse | 0.050953 |
| pose_auc_5 | 7.384615 |
| rot_err_deg | 2.837423 |
| center_err | 16.634313 |
| rot_auc_30 | 56.307692 |
