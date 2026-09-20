# vggt-1b reconstruction comparison — scene `courtyard`, window [5, np.int64(0)], 518x336

depth = the model's dedicated depth head (z-depth), scale-aligned to the GT avg_dis factor (repo protocol);
pose diff = cpp vs torch (mean over views, quat->se3).

| metric | torch | f16 | q8_0 | q5_K |
|--------|--------|--------|--------|--------|
| AbsRel | 0.021158 | 0.020973 | 0.021542 | 0.015032 |
| SqRel | 0.001089 | 0.001073 | 0.001129 | 0.000532 |
| RMSE | 0.023145 | 0.023033 | 0.023599 | 0.019203 |
| RMSE-log | 0.058814 | 0.058333 | 0.060094 | 0.029284 |
| d1 | 0.982500 | 0.982911 | 0.982249 | 0.996840 |
| accuracy | 0.008486 | 0.008467 | 0.008656 | 0.008258 |
| completeness | 0.018466 | 0.018417 | 0.018718 | 0.017566 |
| chamfer | 0.013476 | 0.013442 | 0.013687 | 0.012912 |
| fscore | 0.974117 | 0.974243 | 0.974060 | 0.978806 |
| rot_deg | nan | 0.013583 | 0.021930 | 0.043346 |
| trans_m | nan | 0.000224 | 0.000301 | 0.001572 |
