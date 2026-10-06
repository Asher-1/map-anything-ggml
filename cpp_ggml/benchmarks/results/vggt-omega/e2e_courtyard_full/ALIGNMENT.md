# Full-scene e2e reconstruction — 38 frames of `courtyard`

- chunking: SINGLE PASS (official semantics), 1 forward(s) per runtime; both sides share the identical chunking + alignment code (align=gt)
- merge: official filters (isfinite + depth_edge rtol=0.03 + conf percentile p20.0), conf-priority voxel=0.02

| pair | points | mean | median | p95 | max |
|---|---|---|---|---|---|
| cpp f16single vs torch f32 | 7,712,674 | 0.02096 | 0.01544 | 0.06136 | 0.18501 |

## GT-anchor Sim3 fits

| chunk | side / shared frames | median res | p95 res | scale |
|---|---|---|---|---|
| 0 | torch frames 0-37 | 0.04281 | 0.08261 | 11.25682 |
| 0 | cpp frames 0-37 | 0.04546 | 0.07264 | 11.34211 |

(gt mode: residual = model camera centers vs GT camera centers after the per-chunk Sim3, in GT meters; scale maps each chunk's own model basis to the metric GT frame — chunk-to-chunk variation of `scale` is the model's cross-chunk scale drift, absorbed here instead of accumulating through a chain.)
