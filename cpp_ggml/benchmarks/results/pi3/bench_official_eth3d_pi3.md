# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x392, arch=pi3, torch f32 vs cpp pi3-f16.gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | torch | cpp |
|--------|--------|--------|
| metric_scale_abs_rel | 0.733483 | 0.733502 |
| pointmaps_abs_rel | 0.065016 | 0.064891 |
| pointmaps_inlier_thres_103 | 0.650914 | 0.650215 |
| z_depth_abs_rel | 0.048546 | 0.048571 |
| z_depth_inlier_thres_103 | 0.522134 | 0.522670 |
| ray_dirs_err_deg | 2.069735 | 2.067295 |
| pose_ate_rmse | 0.017195 | 0.017142 |
| pose_auc_5 | 56.615385 | 57.076923 |
| rot_err_deg | 1.815298 | 1.709458 |
| center_err | 5.511334 | 5.466080 |
| rot_auc_30 | 86.333333 | 86.410256 |
