# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 512x336, arch=vggt_omega, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.761842 |
| pointmaps_abs_rel | 0.030519 |
| pointmaps_inlier_thres_103 | 0.851042 |
| z_depth_abs_rel | 0.020955 |
| z_depth_inlier_thres_103 | 0.794225 |
| ray_dirs_err_deg | 1.003297 |
| pose_ate_rmse | 0.006475 |
| pose_auc_5 | 77.076923 |
| rot_err_deg | 0.731755 |
| center_err | 1.769836 |
| rot_auc_30 | 95.076923 |
