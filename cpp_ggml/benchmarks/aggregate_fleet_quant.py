#!/usr/bin/env python3
"""Aggregate the per-model per-quant 130-set benches into a fleet table.

Reads benchmarks/results/<arch>/bench_official_eth3d_<arch>_<q>.md for the
q6_K/q8_0/q5_K tiers and prints, per model, the official-protocol metrics
side by side with a verdict line (q6_K vs q8_0 vs q5_K on depth/ATE/AUC5).
Used for the fleet-wide q6_K confirmation (2026-10-01).
"""
import re
from pathlib import Path

CPP = Path(__file__).resolve().parent.parent
KEYS = ["pointmaps_abs_rel", "z_depth_abs_rel", "pose_ate_rmse",
        "pose_auc_5", "rot_err_deg", "metric_scale_abs_rel"]
MODELS = [("pi3", "pi3"), ("pi3x", "pi3x"),
          ("mapanything", "mapanything"),
          ("vggt", "vggt-1b"), ("dust3r", "dust3r")]


def read_md(path):
    vals = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            try:
                vals[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return vals


def main():
    print("# Fleet-wide q6_K confirmation — official ETH3D 130 sets, "
          "cpp-only, 2026-10-01\n")
    print("Same protocol as the two-sided benches (seed 777, 10 random walks "
          "per scene); gates 15/15 (gate_matrix_q6k_fleet.log).\n")
    for arch, stem in MODELS:
        rows = {}
        for q in ("q6_K", "q8_0", "q5_K"):
            p = CPP / f"benchmarks/results/{stem}/bench_official_eth3d_{arch}_{q}.md"
            rows[q] = read_md(p) if p.exists() else None
        print(f"\n===== {arch} =====")
        print("| metric | " + " | ".join(rows) + " |")
        print("|---" * (len(rows) + 1) + "|")
        for k in KEYS:
            cells = []
            for q in rows:
                v = rows[q].get(k) if rows[q] else None
                cells.append(f"{v:.6f}" if v is not None else "-")
            print(f"| {k} | " + " | ".join(cells) + " |")
        if not all(rows.values()):
            print("verdict: incomplete (missing md files)")
            continue
        (d6, d8, d5) = (rows[q]["z_depth_abs_rel"] for q in rows)
        (a6, a8, a5) = (rows[q]["pose_ate_rmse"] for q in rows)
        (u6, u8, u5) = (rows[q]["pose_auc_5"] for q in rows)
        print(f"verdict: depth q6_K {'<' if d6 < d8 else '>'} q8_0, "
              f"{'<' if d6 < d5 else '>'} q5_K | "
              f"ATE q6_K {'<' if a6 < a8 else '>'} q8_0, "
              f"{'<' if a6 < a5 else '>'} q5_K | "
              f"AUC5 q6_K {'>' if u6 > u8 else '<'} q8_0, "
              f"{'>' if u6 > u5 else '<'} q5_K")


if __name__ == "__main__":
    main()
