# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=pi3x, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | torch | cpp |
|--------|--------|--------|
| metric_scale_abs_rel | 0.250733 | 0.251058 |
| pointmaps_abs_rel | 0.053340 | 0.053448 |
| pointmaps_inlier_thres_103 | 0.718657 | 0.717321 |
| z_depth_abs_rel | 0.036922 | 0.036927 |
| z_depth_inlier_thres_103 | 0.631935 | 0.631366 |
| ray_dirs_err_deg | 0.839664 | 0.848328 |
| pose_ate_rmse | 0.016232 | 0.016280 |
| pose_auc_5 | 66.923077 | 67.076923 |
| rot_err_deg | 2.108082 | 2.112673 |
| center_err | 3.665609 | 3.659379 |
| rot_auc_30 | 90.282051 | 90.358974 |
