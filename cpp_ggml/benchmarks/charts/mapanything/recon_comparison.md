# mapanything reconstruction comparison — scene `courtyard`, window [5, np.int64(0)], 518x336

depth = local_points[..., 2] (camera z, metric), scale-aligned
to the GT avg_dis factor (repo protocol); pose diff = cpp vs
torch (mean over views, metric frames).

| metric | torch | f16 | q8_0 | q6_K |
|--------|--------|--------|--------|--------|
| AbsRel | 0.077106 | 0.077170 | 0.077287 | 0.077505 |
| SqRel | 0.006987 | 0.006997 | 0.007016 | 0.007045 |
| RMSE | 0.078531 | 0.078588 | 0.078683 | 0.078817 |
| RMSE-log | 0.091711 | 0.091808 | 0.091942 | 0.092170 |
| d1 | 0.990103 | 0.990013 | 0.989984 | 0.989798 |
| accuracy | 0.046360 | 0.045456 | 0.045500 | 0.045655 |
| completeness | 0.060051 | 0.060058 | 0.060126 | 0.060235 |
| chamfer | 0.053205 | 0.052757 | 0.052813 | 0.052945 |
| fscore | 0.587746 | 0.591175 | 0.591259 | 0.590351 |
| rot_deg | nan | 0.022276 | 0.009958 | 0.029833 |
| trans_m | nan | 0.009242 | 0.005009 | 0.012942 |
