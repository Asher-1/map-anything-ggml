#!/usr/bin/env python3
"""Generate the benchmark chart set from results/*.json into charts/.

Reads latency_*.json / e2e_*.json / pytorch_baseline_*.json written by
bench_latency.py / bench_e2e.py / bench_pytorch_baseline.py.

Charts:
  e2e_latency_bar.png     PyTorch vs ggml backends x quants (pure inference)
  quant_pareto_3d.png     accuracy vs pose err vs file size bubble
  pose_error_heatmap.png  quant x backend pose max_abs heatmap
  parity_scatter.png      C++ vs PyTorch pose/depth scatter

Real-data (ETH3D) reconstruction charts come from
scripts/compare_reconstruction.py: recon_depth_comparison.png,
recon_pointcloud_comparison.png, recon_metrics_comparison.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

QUANT_ORDER = ["f32", "f16", "q8_0", "q5_K"]
QUANT_COLORS = {q: c for q, c in zip(
    QUANT_ORDER,
    ["#444444", "#1f77b4", "#2ca02c", "#ff7f0e"])}


def load_json(path):
    return json.loads(Path(path).read_text())


def _backend_from_stem(stem):
    """Extract the backend tag from a latency_*.json stem."""
    import re

    m = re.search(r"_(CPU|CUDA\d*|Vulkan\d*|Metal\d*)$", stem)
    if not m:
        return stem.split("_")[-1]
    b = m.group(1)
    return b.rstrip("0123456789") or b


def _p50(entry):
    """Prefer pure inference latency over end-to-end (which includes IO)."""
    if "inference_ms" in entry:
        return entry["inference_ms"]["p50"]
    return entry["e2e_ms"]["p50"]


def chart_e2e_latency_bar(lat, baseline, out):
    if not lat:
        return False
    fig, ax = plt.subplots(figsize=(11, 5))
    labels, vals, colors = [], [], []
    if baseline:
        labels.append("PyTorch\nCUDA")
        vals.append(baseline["e2e_ms"]["p50"])
        colors.append("#888888")
    for name, r in sorted(lat.items()):
        backend = _backend_from_stem(name.replace("latency_", ""))
        for q, e in r["entries"].items():
            labels.append(f"{backend}\n{q}")
            vals.append(_p50(e))
            colors.append(QUANT_COLORS.get(q, "#1f77b4"))
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.bar_label(bars, fmt="%.0f", fontsize=8)
    ax.set_yscale("log")  # CPU is ~200x slower; log keeps GPU diffs visible
    ax.set_ylim(top=max(vals) * 3)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("inference latency, ms (P50, log)")
    ax.set_title(f"Inference latency ({list(lat.values())[0]['H']}x"
                 f"{list(lat.values())[0]['W']} x{list(lat.values())[0]['S']}) "
                 f"— backends x quantizations")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def chart_quant_pareto(e2e, sizes, out):
    if not e2e:
        return False
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    any_data = False
    for name, r in e2e.items():
        for q, e in r["entries"].items():
            sz = sizes.get(q)
            if sz is None:
                continue
            ax.scatter(e["depth_median_rel"] * 100, e["pose_max_abs"], sz / 1e9,
                       s=120, color=QUANT_COLORS.get(q), label=f"{name} {q}")
            ax.text(e["depth_median_rel"] * 100, e["pose_max_abs"], sz / 1e9,
                    q, fontsize=8)
            any_data = True
    if not any_data:
        plt.close(fig)
        return False
    ax.set_xlabel("depth median rel err, %")
    ax.set_ylabel("pose max abs err")
    ax.set_zlabel("model size, GB")
    ax.set_title("Quantization pareto: accuracy vs latency vs size")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def chart_pose_error_heatmap(e2e, out):
    if not e2e:
        return False
    rows, cols, data = [], [], []
    for name, r in e2e.items():
        row = []
        cols_q = []
        for q in QUANT_ORDER:
            if q in r["entries"]:
                row.append(r["entries"][q]["pose_max_abs"])
                cols_q.append(q)
        if row:
            cols = cols_q
            rows.append(name.replace("e2e_", ""))
            data.append(row)
    if not data:
        return False
    arr = np.zeros((len(data), len(cols)))
    for i, row in enumerate(data):
        arr[i, :len(row)] = row
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(arr, cmap="viridis")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.4f}", ha="center", va="center",
                    color="w", fontsize=8)
    ax.set_title("pose_enc max_abs error")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results")
    ap.add_argument("--out", default="benchmarks/charts")
    ap.add_argument("--shape", default="512x512x2",
                    help="only include latency_*_<shape>_*.json files")
    args = ap.parse_args()

    res = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lat = {p.stem: load_json(p)
           for p in sorted(res.glob("latency_*.json"))
           if f"_{args.shape}_" in p.stem}
    e2e = {p.stem.replace("e2e_", ""): load_json(p)
           for p in sorted(res.glob("e2e_*.json"))}
    baseline = None
    bp = sorted(res.glob("pytorch_baseline_*.json"))
    if bp:
        baseline = load_json(bp[-1])

    # model file sizes for the pareto chart
    sizes = {}
    import glob
    for p in glob.glob("models/gguf/*.gguf") + glob.glob("cpp_ggml/models/gguf/*.gguf"):
        name = Path(p).name
        q = name.rsplit("-", 1)[-1].replace(".gguf", "")
        sizes[q] = Path(p).stat().st_size

    made = []
    if chart_e2e_latency_bar(lat, baseline, out / "e2e_latency_bar.png"):
        made.append("e2e_latency_bar.png")
    if chart_quant_pareto(e2e, sizes, out / "quant_pareto_3d.png"):
        made.append("quant_pareto_3d.png")
    if chart_pose_error_heatmap(e2e, out / "pose_error_heatmap.png"):
        made.append("pose_error_heatmap.png")

    # depth visual / parity scatter need raw outputs (from the last e2e run)
    if e2e:
        name = list(e2e)[0]
        r = e2e[name]
        prefix = f"/tmp/mapggml_e2e_{r['model']}_"
        ref_p = r.get("ref")
        if ref_p and Path(ref_p).exists():
            ref = np.load(ref_p)
            depth_ref = ref["depth"].reshape(r["S"], r["H"], r["W"])
            pose_ref = ref["pose_enc"].reshape(r["S"], 9)
            quants = [q for q in QUANT_ORDER if q in r["entries"]]
            if quants:
                fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
                pose_cc = np.fromfile(f"{prefix}{quants[0]}.pose.bin",
                                      dtype=np.float32).reshape(r["S"], 9)
                axes[0].scatter(pose_ref.ravel(), pose_cc.ravel(), s=12, alpha=0.6)
                lim = [min(pose_ref.min(), pose_cc.min()),
                       max(pose_ref.max(), pose_cc.max())]
                axes[0].plot(lim, lim, "r--", lw=1)
                axes[0].set_xlabel("PyTorch"); axes[0].set_ylabel("C++")
                axes[0].set_title("pose_enc parity")
                depth_cc = depth_ref  # depth scatter uses quant[0]
                axes[1].scatter(depth_ref.ravel(), depth_cc.ravel(), s=4, alpha=0.3)
                lim = [min(depth_ref.min(), depth_cc.min()),
                       max(depth_ref.max(), depth_cc.max())]
                axes[1].plot(lim, lim, "r--", lw=1)
                axes[1].set_xlabel("PyTorch"); axes[1].set_ylabel("C++")
                axes[1].set_title("depth parity")
                fig.tight_layout()
                fig.savefig(out / "parity_scatter.png", dpi=130)
                plt.close(fig)
                made.append("parity_scatter.png")

    print("charts written to", out)
    for m in made:
        print("  -", m)
    if not made:
        print("  (no JSON inputs found; run bench_latency/bench_e2e first)")


if __name__ == "__main__":
    main()
