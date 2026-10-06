# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=mapanything, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.162189 |
| pointmaps_abs_rel | 0.055484 |
| pointmaps_inlier_thres_103 | 0.616372 |
| z_depth_abs_rel | 0.047245 |
| z_depth_inlier_thres_103 | 0.527435 |
| ray_dirs_err_deg | 1.116609 |
| pose_ate_rmse | 0.012431 |
| pose_auc_5 | 57.692308 |
| rot_err_deg | 0.899451 |
| center_err | 4.694857 |
| rot_auc_30 | 86.923077 |
