# pi3x reconstruction comparison — scene `courtyard`, window [5, np.int64(0)], 518x336

depth = local_points[..., 2] (camera z, metric), scale-aligned
to the GT avg_dis factor (repo protocol); pose diff = cpp vs
torch (mean over views, metric frames).

| metric | torch | f16 | q8_0 | q5_K |
|--------|--------|--------|--------|--------|
| AbsRel | 0.015779 | 0.015774 | 0.016317 | 0.017492 |
| SqRel | 0.000546 | 0.000546 | 0.000570 | 0.000608 |
| RMSE | 0.020101 | 0.020096 | 0.020521 | 0.021182 |
| RMSE-log | 0.026027 | 0.026072 | 0.026776 | 0.027913 |
| d1 | 0.999368 | 0.999368 | 0.999368 | 0.999368 |
| accuracy | 0.007980 | 0.007841 | 0.008011 | 0.008501 |
| completeness | 0.015648 | 0.015628 | 0.015783 | 0.016326 |
| chamfer | 0.011814 | 0.011735 | 0.011897 | 0.012413 |
| fscore | 0.981632 | 0.981998 | 0.981998 | 0.982204 |
| rot_deg | nan | 0.027976 | 0.088310 | 0.904626 |
| trans_m | nan | 0.005162 | 0.031622 | 0.053017 |
