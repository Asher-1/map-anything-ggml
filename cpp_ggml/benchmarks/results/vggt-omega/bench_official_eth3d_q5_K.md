# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 512x336)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.759581 |
| pointmaps_abs_rel | 0.032789 |
| pointmaps_inlier_thres_103 | 0.835230 |
| z_depth_abs_rel | 0.022053 |
| z_depth_inlier_thres_103 | 0.769325 |
| ray_dirs_err_deg | 1.056539 |
| pose_ate_rmse | 0.007117 |
| pose_auc_5 | 71.692308 |
| rot_err_deg | 0.879015 |
| center_err | 2.119118 |
| rot_auc_30 | 94.076923 |
