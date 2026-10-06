# Official MapAnything ETH3D protocol (random-walk sampling, seed 777, 2 views, 518x336, arch=vggt, torch f32 vs cpp gguf)

sets evaluated: 130 (10 per scene x 13 scenes)

Official reference (reproduction.md, retrained ckpt, 2-100 views average): AUC 79.53 / ATE 0.009985 / Point Abs 0.026265 / Depth Abs 0.020429

| metric | cpp |
|--------|--------|
| metric_scale_abs_rel | 0.788386 |
| pointmaps_abs_rel | 0.071394 |
| pointmaps_inlier_thres_103 | 0.620495 |
| z_depth_abs_rel | 0.051085 |
| z_depth_inlier_thres_103 | 0.532652 |
| ray_dirs_err_deg | 0.837569 |
| pose_ate_rmse | 0.022965 |
| pose_auc_5 | 63.076923 |
| rot_err_deg | 2.192246 |
| center_err | 4.458269 |
| rot_auc_30 | 88.743590 |
