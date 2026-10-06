# Fleet-wide q6_K confirmation — official ETH3D 130 sets, cpp-only, 2026-10-01

Same protocol as the two-sided benches (seed 777, 10 random walks per scene); gates 15/15 (gate_matrix_q6k_fleet.log).

Note (later 2026-10-01): each model now ships exactly ONE measured K
tier. q6_K kept for omega/pi3x/mapanything/dust3r; q5_K kept for pi3
(best depth/rot/ATE/pointmaps) and vggt-1b (best AUC5); the pi3/vggt-1b
q6_K columns above record why their q6_K was removed (no best metric /
weakest tier on 5 of 7 metrics).


===== pi3 =====
| metric | q6_K | q8_0 | q5_K |
|---|---|---|---|
| pointmaps_abs_rel | 0.069685 | 0.069590 | 0.065244 |
| z_depth_abs_rel | 0.048355 | 0.050457 | 0.046483 |
| pose_ate_rmse | 0.018509 | 0.018389 | 0.018289 |
| pose_auc_5 | 51.692308 | 52.923077 | 50.461538 |
| rot_err_deg | 1.577774 | 1.341748 | 1.086058 |
| metric_scale_abs_rel | 0.733665 | 0.733509 | 0.733816 |
verdict: depth q6_K < q8_0, > q5_K | ATE q6_K > q8_0, > q5_K | AUC5 q6_K < q8_0, > q5_K

===== pi3x =====
| metric | q6_K | q8_0 | q5_K |
|---|---|---|---|
| pointmaps_abs_rel | 0.053740 | 0.053749 | 0.054870 |
| z_depth_abs_rel | 0.036885 | 0.037098 | 0.036450 |
| pose_ate_rmse | 0.016311 | 0.016243 | 0.016303 |
| pose_auc_5 | 66.153846 | 67.384615 | 62.461538 |
| rot_err_deg | 2.120841 | 2.121493 | 2.294897 |
| metric_scale_abs_rel | 0.251801 | 0.252437 | 0.243974 |
verdict: depth q6_K < q8_0, > q5_K | ATE q6_K > q8_0, > q5_K | AUC5 q6_K < q8_0, > q5_K

===== mapanything =====
| metric | q6_K | q8_0 | q5_K |
|---|---|---|---|
| pointmaps_abs_rel | 0.055542 | 0.055484 | 0.056155 |
| z_depth_abs_rel | 0.047034 | 0.047245 | 0.047824 |
| pose_ate_rmse | 0.012504 | 0.012431 | 0.012566 |
| pose_auc_5 | 56.307692 | 57.692308 | 56.000000 |
| rot_err_deg | 0.891136 | 0.899451 | 0.895226 |
| metric_scale_abs_rel | 0.164076 | 0.162189 | 0.162483 |
verdict: depth q6_K < q8_0, < q5_K | ATE q6_K > q8_0, < q5_K | AUC5 q6_K < q8_0, > q5_K

===== vggt =====
| metric | q6_K | q8_0 | q5_K |
|---|---|---|---|
| pointmaps_abs_rel | 0.074796 | 0.071394 | 0.074521 |
| z_depth_abs_rel | 0.052635 | 0.051085 | 0.052132 |
| pose_ate_rmse | 0.024855 | 0.022965 | 0.024029 |
| pose_auc_5 | 62.923077 | 63.076923 | 63.692308 |
| rot_err_deg | 2.459645 | 2.192246 | 2.523399 |
| metric_scale_abs_rel | 0.787963 | 0.788386 | 0.788555 |
verdict: depth q6_K > q8_0, > q5_K | ATE q6_K > q8_0, > q5_K | AUC5 q6_K < q8_0, < q5_K

===== dust3r =====
| metric | q6_K | q8_0 | q5_K |
|---|---|---|---|
| pointmaps_abs_rel | 0.101118 | 0.100995 | 0.101499 |
| z_depth_abs_rel | 0.102975 | 0.103174 | 0.103523 |
| pose_ate_rmse | 0.050718 | 0.050953 | 0.051103 |
| pose_auc_5 | 8.000000 | 7.384615 | 7.538462 |
| rot_err_deg | 2.806158 | 2.837423 | 2.844861 |
| metric_scale_abs_rel | 0.897447 | 0.897440 | 0.897507 |
verdict: depth q6_K < q8_0, < q5_K | ATE q6_K < q8_0, < q5_K | AUC5 q6_K > q8_0, > q5_K
