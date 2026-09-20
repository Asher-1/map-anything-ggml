# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 512x336)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | torch | cpp |
|--------|--------|--------|
| metric_scale_abs_rel | 0.788022 | 0.788020 |
| pointmaps_abs_rel | 0.067441 | 0.067684 |
| pointmaps_inlier_thres_103 | 0.663644 | 0.662515 |
| z_depth_abs_rel | 0.054077 | 0.054395 |
| z_depth_inlier_thres_103 | 0.578923 | 0.576816 |
| ray_dirs_err_deg | 0.776690 | 0.778217 |
| pose_ate_rmse | 0.020764 | 0.020857 |
| pose_auc_5 | 64.000000 | 63.384615 |
| rot_err_deg | 2.275437 | 2.239202 |
| center_err | 3.655348 | 3.665375 |
| rot_auc_30 | 89.256410 | 89.282051 |
