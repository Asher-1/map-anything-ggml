# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 512x336, arch=vggt_omega, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | torch | cpp |
|--------|--------|--------|
| metric_scale_abs_rel | 0.761655 | 0.761808 |
| pointmaps_abs_rel | 0.030160 | 0.030178 |
| pointmaps_inlier_thres_103 | 0.853239 | 0.852966 |
| z_depth_abs_rel | 0.020752 | 0.020924 |
| z_depth_inlier_thres_103 | 0.797842 | 0.794081 |
| ray_dirs_err_deg | 0.993159 | 0.994794 |
| pose_ate_rmse | 0.006549 | 0.006618 |
| pose_auc_5 | 79.230769 | 79.076923 |
| rot_err_deg | 0.556370 | 0.621498 |
| center_err | 1.655104 | 1.645653 |
| rot_auc_30 | 95.692308 | 95.666667 |
