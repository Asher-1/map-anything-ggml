# pi3 reconstruction comparison — scene `courtyard`, window [5, np.int64(0)], 518x336

depth = local_points[..., 2] (camera z, metric), scale-aligned
to the GT avg_dis factor (repo protocol); pose diff = cpp vs
torch (mean over views, metric frames).

| metric | torch | f16 | q8_0 | q5_K |
|--------|--------|--------|--------|--------|
| AbsRel | 0.013730 | 0.013543 | 0.014092 | 0.014039 |
| SqRel | 0.000485 | 0.000482 | 0.000500 | 0.000497 |
| RMSE | 0.018420 | 0.018365 | 0.018750 | 0.018718 |
| RMSE-log | 0.023596 | 0.023543 | 0.023974 | 0.024131 |
| d1 | 0.999174 | 0.999159 | 0.999166 | 0.999069 |
| accuracy | 0.007360 | 0.007160 | 0.007409 | 0.006304 |
| completeness | 0.014429 | 0.014282 | 0.014578 | 0.013763 |
| chamfer | 0.010894 | 0.010721 | 0.010994 | 0.010034 |
| fscore | 0.979496 | 0.979527 | 0.979416 | 0.979843 |
| rot_deg | nan | 0.000000 | 0.120867 | 1.208889 |
| trans_m | nan | 0.002180 | 0.004678 | 0.015320 |
