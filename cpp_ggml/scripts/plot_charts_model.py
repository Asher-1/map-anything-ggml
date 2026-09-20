#!/usr/bin/env python3
"""Per-model chart generator — ONE script for all supported families,
replacing the pi3x-only draft (its omega-style latency chart, pose
heatmap, quant pareto and parity scatter are arch-parameterized here).

  python3 scripts/plot_charts_model.py --arch pi3x  \
      --gate-log /tmp/gate_pi3x_matrix2.log \
      --torch-prefix /tmp/pi3x_smoke/torch --cpp-prefix /tmp/gate_cpu_check
  python3 scripts/plot_charts_model.py --arch pi3    \
      --gate-log /tmp/gate_matrix_rerun.log \
      --torch-prefix /tmp/pi3_cal/torch    --cpp-prefix /tmp/pi3_cal/cpp
  python3 scripts/plot_charts_model.py --arch vggt   \
      --gate-log /tmp/gate_matrix_rerun.log \
      --torch-prefix /tmp/vggt1b_torch     --cpp-prefix /tmp/gate_vggt_check

Contracts: pi3/pi3x -> pose16 (S,16) + local_points (S,H,W,3) + conf +
scale; vggt -> pose (S,9) + depth (S,H,W) + depth_conf.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CPP = Path(__file__).resolve().parent.parent
QUANT_ORDER = ["f32", "f16", "q8_0", "q5_K"]
QCOLOR = {"f32": "#8c564b", "f16": "#ff7f0e", "q8_0": "#2ca02c",
          "q5_K": "#d62728"}
GGUF_RES = {"pi3": 518, "pi3x": 518, "vggt": 518, "mapanything": 518,
            "dust3r": 512}
GGUF_STEM = {"pi3": "pi3", "pi3x": "pi3x", "vggt": "vggt-1b",
             "mapanything": "mapanything", "dust3r": "dust3r"}


def load_json(p):
    return json.loads(Path(p).read_text())


def chart_latency(lat, torch_refs, out, arch):
    if not lat:
        return False
    fig, ax = plt.subplots(figsize=(11, 5))
    labels, vals, colors = [], [], []
    for label, ms in torch_refs.items():
        labels.append(f"PyTorch\n{label}")
        vals.append(ms)
        colors.append("#888888")
    for b in ("CUDA", "Vulkan", "CPU"):
        for q in QUANT_ORDER:
            if lat.get(b, {}).get(q) is not None:
                labels.append(f"{b}\n{q}")
                vals.append(lat[b][q])
                colors.append(QCOLOR.get(q, "#1f77b4"))
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.bar_label(bars, fmt="%.0f", fontsize=8)
    ax.set_yscale("log")  # CPU is ~1000x slower; log keeps GPU visible
    ax.set_ylim(top=max(vals) * 3)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("inference latency, ms (P50, log)")
    ax.set_title(f"{arch} inference latency — official PyTorch vs cpp_ggml "
                 f"backends x quantizations ({GGUF_RES[arch]}x{GGUF_RES[arch]} S=2)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def parse_gate_log(path):
    log = Path(path).read_text()
    blocks = re.findall(
        r"== (\S+) (\S+) (\S+) ==\n((?:  .*\n)+?)  GATE: (\w+)", log)
    out = {}
    for arch, q, build, body, verdict in blocks:
        m = {k: (float(mx), float(mr)) for k, mx, mr in re.findall(
            r"  (\S+)\s+max ([\d.e-]+)\s+med_rel ([\d.e-]+)", body)}
        m["verdict"] = (None, verdict)
        # keep the arch: a combined-matrix log would otherwise collide
        # across families on the same (quant, backend) key
        out[(arch, q, build.replace("build-", ""))] = m
    return out


def chart_pose_heatmap(gate, pose_key, out, arch):
    quants = [q for q in QUANT_ORDER if any(q == k[0] for k in gate)]
    backends = [b for b in ("cpu", "cuda", "vulkan")
                if any(k[1] == b for k in gate)]
    M = np.full((len(quants), len(backends)), np.nan)
    for i, q in enumerate(quants):
        for j, b in enumerate(backends):
            if (q, b) in gate:
                M[i, j] = gate[(q, b)][pose_key][0]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(backends)), backends)
    ax.set_yticks(range(len(quants)), quants)
    for i in range(len(quants)):
        for j in range(len(backends)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                        fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{arch} {pose_key} max_abs — quant x backend")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def chart_pareto(gate, acc_key, sizes, out, arch):
    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_subplot(projection="3d")
    for q in QUANT_ORDER:
        cells = [(b, gate[(q, b)]) for b in ("cpu", "cuda", "vulkan")
                 if (q, b) in gate]
        if not cells:
            continue
        xs = [sizes.get(q, np.nan) / 1e9 for _, _ in cells]
        ys = [v[acc_key][1] for _, v in cells]
        zs = [v[acc_key][0] for _, v in cells]
        ax.scatter(xs, ys, zs, s=60, color=QCOLOR[q], label=q,
                   depthshade=True)
        for b, v in cells:
            ax.text(sizes.get(q, 0) / 1e9, v[acc_key][1], v[acc_key][0],
                    b, fontsize=6)
    ax.set_xlabel("gguf size (GB)")
    ax.set_ylabel(f"{acc_key} med_rel")
    ax.set_zlabel(f"{acc_key} max_abs")
    ax.set_title(f"{arch} quant pareto: size vs parity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def chart_parity(torch_prefix, cpp_prefix, spec_t, spec_c, out, arch):
    def lt(p, shape=None):
        a = np.fromfile(p, dtype=np.float32)
        return a.reshape(shape) if shape else a
    def ref_path(prefix, name):
        # tolerate dir-style ("/x/spy2/") and prefix-style ("/x/torch")
        # references in one helper
        for cand in (f"{prefix}{name}.bin", f"{prefix}/{name}.bin",
                     f"{prefix}.{name}.bin"):
            if Path(cand).exists():
                return cand
        raise FileNotFoundError(f"{prefix}+{name}.bin")

    panels = []
    for (tf, ts), (cf, cs) in zip(spec_t, spec_c):
        panels.append((lt(ref_path(torch_prefix, tf), ts),
                       lt(ref_path(cpp_prefix, cf), cs), tf))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.8))
    if len(panels) == 1:
        axes = [axes]
    for ax, (t, c, name) in zip(axes, panels):
        tt, cc = t.reshape(-1), c.reshape(-1)   # 2-D arrays: ravel first,
        idx = np.random.default_rng(0).integers(0, tt.size,   # else scatter
                                                 min(tt.size, 60000))  # indexes rows
        ax.scatter(tt[idx], cc[idx], s=3, alpha=0.35)
        lim = [min(tt.min(), cc.min()), max(tt.max(), cc.max())]
        ax.plot(lim, lim, "r--", lw=1)
        r = np.corrcoef(tt, cc)[0, 1]
        ax.set_xlabel("PyTorch (official)")
        ax.set_ylabel("C++ ggml")
        ax.set_title(f"{name} parity  (corr {r:.5f})")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{arch} (cpp f16 CUDA) vs official torch f32 — fixed "
                 f"frames518.bin")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# per family: (parity panels, gate-metric row keys)
#   parity panels: (torch file, shape), (cpp file, shape) — the file name
#   is appended to the prefix with a separator chosen per prefix style
#   (dir-style "/x/" or prefix-style "/x.out");
#   gate keys: (pose max-abs row, accuracy med_rel row) in cmp_gate_matrix.
SPECS = {
    "pi3": (
        [("pose16", (2, 16)), ("local_points", (2, 518, 518, 3))],
        [("pose", (2, 16)), ("local_points", (2, 518, 518, 3))],
        "pose16", "local_points"),
    "pi3x": (
        [("pose16", (2, 16)), ("local_points", (2, 518, 518, 3))],
        [("pose", (2, 16)), ("local_points", (2, 518, 518, 3))],
        "pose16", "local_points"),
    "vggt": (
        [("pose", (2, 9)), ("depth", (2, 518, 518))],
        [("pose", (2, 9)), ("depth", (2, 518, 518))],
        "pose_enc", "depth"),
    # mapanything torch ref stores per-view files (out_depth0/1.bin, ...);
    # main() merges them into depth_all/pts3d_all before the parity chart
    "mapanything": (
        [("depth_all", (2, 518, 518)), ("pts3d_all", (2, 518, 518, 3))],
        [("depth", (2, 518, 518)), ("points", (2, 518, 518, 3))],
        "pose_raw", "depth"),
    # dust3r: no pose at all — the heatmap/pareto keys reuse local_points;
    # the spy stores the two views as separate files (view1 self-view,
    # view2 other-view, both in view1's frame), merged by main() like
    # mapanything's; gate runs at 512x512 (patch 16)
    "dust3r": (
        [("pts3d_all", (2, 512, 512, 3)), ("conf_all", (2, 512, 512))],
        [("local_points", (2, 512, 512, 3)), ("conf", (2, 512, 512))],
        "local_points", "local_points"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["pi3", "pi3x", "vggt", "mapanything", "dust3r"],
                    help="results/charts directory stem (vggt = vggt-1b; "
                         "dust3r runs at 512). vggt-omega is NOT handled "
                         "here — it uses benchmarks/plot_charts.py with the "
                         "older pytorch_baseline/e2e JSON layout")
    ap.add_argument("--results", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--gate-log", default="",
                    help="gate-matrix log for THIS arch (parsed for the "
                         "heatmap/pareto; persisted to gate_summary.json). "
                         "Omit to refresh ONLY the latency bar (keeps the "
                         "existing heatmap/pareto/gate_summary untouched)")
    ap.add_argument("--torch-prefix", default="",
                    help="torch ref dump prefix for the parity scatter")
    ap.add_argument("--cpp-prefix", default="",
                    help="cpp f16 output prefix for the parity scatter")
    args = ap.parse_args()
    arch = args.arch
    # directory name follows the gguf stem (vggt -> vggt-1b) so results and
    # charts land in the same per-model folder the README indexes
    dname = GGUF_STEM[arch]
    res = Path(args.results or CPP / "benchmarks/results" / dname)
    out = Path(args.out or CPP / "benchmarks/charts" / dname)
    out.mkdir(parents=True, exist_ok=True)
    stem = dname
    pose_key, acc_key = SPECS[arch][2], SPECS[arch][3]

    lat = {}
    for p in sorted(res.glob(f"latency_{stem}_*.json")):
        r = load_json(p)
        for q, e in r["entries"].items():
            # CLI records ggml's native backend names (CUDA0/Vulkan0);
            # normalize so chart_latency's ("CUDA","Vulkan","CPU") loop
            # sees them (CUDA0 != CUDA silently dropped every GPU bar)
            b = e["samples"][0]["backend"]
            b = {"CUDA0": "CUDA", "Vulkan0": "Vulkan"}.get(b, b)
            lat.setdefault(b, {})[q] = \
                e["inference_ms"]["p50"]
    torch_refs = {}
    for p in sorted(res.glob("bench_torch_*.json")):
        r = load_json(p)
        lbl = p.stem.replace("bench_torch_", "").replace(arch, "").strip("_")
        lbl = (lbl.replace("cuda_tf32", "fp32+TF32 CUDA")
                  .replace("cuda", "fp32 CUDA")
                  .replace("cpu", "fp32 CPU").strip("_")) or "fp32 CUDA"
        torch_refs[lbl] = r["infer_ms_p50"]

    made = []
    if lat and chart_latency(lat, torch_refs, out / "e2e_latency_bar.png",
                             arch):
        made.append("e2e_latency_bar.png")

    if not args.gate_log:
        # latency-only refresh: keep the existing gate charts/summary
        print("refreshed:", made or "nothing (no latency jsons found)")
        return

    gate = parse_gate_log(args.gate_log)
    gate = {(q, b): v for (a, q, b), v in gate.items()
            if a == arch and q in QUANT_ORDER
            and b in ("cpu", "cuda", "vulkan")}
    if not gate:
        known = sorted({a for (a, q, b) in parse_gate_log(args.gate_log)})
        raise SystemExit(f"no gate cells for arch '{arch}' in "
                         f"{args.gate_log} (arches present: {known})")
    if gate:
        summary = {f"{q}|{b}": {"verdict": v["verdict"][1],
                                **{k: v[k][1] for k in v
                                   if k not in ("verdict",)}}
                   for (q, b), v in gate.items()}
        (out / "gate_summary.json").write_text(json.dumps(summary, indent=1))
    if gate and chart_pose_heatmap(gate, pose_key,
                                   out / "pose_error_heatmap.png", arch):
        made.append("pose_error_heatmap.png")
    sizes = {}
    for q in QUANT_ORDER:
        p = CPP / "models/gguf" / f"{stem}-{q}.gguf"
        if p.exists():
            sizes[q] = p.stat().st_size
    if gate and chart_pareto(gate, acc_key, sizes,
                             out / "quant_pareto_3d.png", arch):
        made.append("quant_pareto_3d.png")

    if args.torch_prefix and args.cpp_prefix:
        if arch == "mapanything":
            # merge the spy's per-view files into depth_all/pts3d_all so the
            # parity chart can read single (S,...) arrays like every family
            td = Path(args.torch_prefix)
            if not (td / "depth_all.bin").exists():
                for name, shape in (("depth", (518, 518)),
                                    ("pts3d", (518, 518, 3))):
                    vs = [np.fromfile(td / f"out_{name}{v}.bin",
                                      dtype=np.float32).reshape(shape)
                          for v in range(2) if (td / f"out_{name}{v}.bin").exists()]
                    if vs:
                        np.stack(vs).tofile(td / f"{name}_all.bin")
        elif arch == "dust3r":
            # same merge pattern: spy per-view files -> *_all (S-first order
            # matching the C++ local_points s0=face1 / s1=head2 layout)
            td = Path(args.torch_prefix)
            if not (td / "pts3d_all.bin").exists():
                np.stack([np.fromfile(td / "pts3d_view1.bin", dtype=np.float32),
                          np.fromfile(td / "pts3d_view2_in_view1.bin",
                                      dtype=np.float32)]) \
                    .tofile(td / "pts3d_all.bin")
            if not (td / "conf_all.bin").exists():
                np.stack([np.fromfile(td / "conf_view1.bin", dtype=np.float32),
                          np.fromfile(td / "conf_view2.bin", dtype=np.float32)]) \
                    .tofile(td / "conf_all.bin")
        spec_t, spec_c, _, _ = SPECS[arch]
        if chart_parity(args.torch_prefix, args.cpp_prefix,
                        spec_t, spec_c, out / "parity_scatter.png", arch):
            made.append("parity_scatter.png")
    print("made:", ", ".join(made) if made else
          "NOTHING (missing inputs?)", "->", out)


if __name__ == "__main__":
    main()
