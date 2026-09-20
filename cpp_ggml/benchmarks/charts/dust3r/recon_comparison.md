# dust3r reconstruction comparison — scene `courtyard`, window [5, np.int64(0)], 512x336

depth = local_points[..., 2] (camera z, metric), scale-aligned
to the GT avg_dis factor (repo protocol); pose diff = cpp vs
torch (mean over views, metric frames).

| metric | torch | f16 | q8_0 | q5_K |
|--------|--------|--------|--------|--------|
| AbsRel | 0.132059 | 0.132052 | 0.132158 | 0.132537 |
| SqRel | 0.014187 | 0.014188 | 0.014180 | 0.014245 |
| RMSE | 0.112869 | 0.112887 | 0.112731 | 0.112919 |
| RMSE-log | 0.131248 | 0.131238 | 0.131393 | 0.131828 |
| d1 | 0.976859 | 0.976905 | 0.975183 | 0.972518 |
| accuracy | 0.077808 | 0.077666 | 0.077512 | 0.077845 |
| completeness | 0.085387 | 0.085414 | 0.085185 | 0.085395 |
| chamfer | 0.081598 | 0.081540 | 0.081348 | 0.081620 |
| fscore | 0.373871 | 0.374023 | 0.373898 | 0.373764 |
| rot_deg | nan | 0.000000 | 0.019782 | 0.013988 |
| trans_m | nan | 0.000028 | 0.000217 | 0.000147 |
