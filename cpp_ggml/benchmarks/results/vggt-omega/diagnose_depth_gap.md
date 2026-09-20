# Depth AbsRel gap diagnosis (torch vs cpp f16, real ETH3D)

Hypothesis: the cpp-vs-torch AbsRel gap concentrates on GT-invalid border pixels (unconstrained outputs); interior pixels agree.

## Erosion sweep (AbsRel after shrinking the valid mask by k px)

| k (px) | PyTorch f32 | C++ ggml f16 | ratio |
|---|---|---|---|
| 0 | 0.01229 | 0.05057 | 4.11x |
| 1 | 0.01023 | 0.04898 | 4.79x |
| 2 | 0.00947 | 0.04857 | 5.13x |
| 4 | 0.00914 | 0.04752 | 5.20x |
| 8 | 0.00890 | 0.05162 | 5.80x |
| 16 | 0.00863 | 0.05044 | 5.84x |

- C++ worst-0.5% error pixels within 2 px of the GT-mask border: **82.7%** (n=2383)

