#!/usr/bin/env python3
"""Gate comparator for the (arch x quant x backend) matrix.

Usage: cmp_gate_matrix.py <arch> <out-prefix> [S H W]
       cmp_gate_matrix.py <arch> <out-prefix> <quant> [ref] [S H W]
       cmp_gate_matrix.py <arch> <out-prefix> <quant> [ref]
Arch keys match the GGUF general.architecture values (vggt-omega gates
through scripts/e2e_gate.sh instead). Reference caches (torch f32, fixed
parity frames):
  pi3          /tmp/pi3_spy2      (spy_torch_pi3.py)
  pi3x         /tmp/pi3x_smoke/torch.*  (spy_torch_pi3x.py)
  mapanything  /tmp/ma_spy        (spy_torch_mapanything.py)
  vggt         /tmp/vggt1b_torch  (dump_torch_vggt.py, pose_enc S,9)
  dust3r       /tmp/mapggml/ref/dust3r  (spy_torch_dust3r.py; 512x512,
               per-view files merged in the s0/s1 CLI contract order)
Prints per-output max/median errors; exits non-zero above thresholds that
scale with the weight quantization level (f32/f16 tight, q8_0 3x, q5_K 6x).
"""
import sys
import numpy as np

arch, prefix = sys.argv[1], sys.argv[2]
S, H, W = 2, 518, 518
q = sys.argv[3] if len(sys.argv) > 3 else "f16"
# optional torch-reference override (arg 4): the directory/prefix the spy
# dump was written to, joined per-family below (dir-style for pi3 and
# mapanything, prefix-style for pi3x and vggt).  Defaults are the historical
# session cache paths so old invocations keep working.
REF = sys.argv[4] if len(sys.argv) > 4 else ""
# optional S/H/W override (arg 5..7) for non-default-view-count gates, e.g.
# the S=4 multi-view parity checks (defaults keep old invocations working)
if len(sys.argv) > 7:
    S, H, W = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
scale = {"f32": 1.0, "f16": 1.0, "q8_0": 3.0, "q5_K": 6.0}.get(q, 1.0)


def load(p, shape=None):
    a = np.fromfile(p, dtype=np.float32)
    return a.reshape(shape) if shape else a


def rep(name, a, b, thr_med=0.010, thr_max=None):
    d = np.abs(a - b)
    rel = d / np.maximum(np.abs(b), 0.05) if name.startswith(("depth", "local", "pts")) else None
    med = float(np.median(rel)) if rel is not None else float(d.mean())
    mx = float(d.max())
    line = f"  {name:16s} max {mx:.5f}  med_rel {med:.5f}  (gate med < {thr_med:g})"
    ok = med < thr_med and (thr_max is None or mx < thr_max)
    print(line + ("" if ok else "  [FAIL]"))
    return ok


ok = True
if arch == "pi3":
    td = REF or "/tmp/pi3_spy2"            # dir-style: <td>/pose16.bin ...
    t = load(f"{td}/pose16.bin", (S, 16))
    c = load(f"{prefix}.pose.bin", (S, 16))
    ok &= rep("pose16", t, c, 0.010 * scale, 0.030 * scale)
    t = load(f"{td}/local_points.bin", (S, H, W, 3))
    c = load(f"{prefix}.local_points.bin", (S, H, W, 3))
    ok &= rep("local_points", t, c, 0.010 * scale)
    t = load(f"{td}/conf.bin", (S, H, W, 1))[..., 0]
    c = load(f"{prefix}.conf.bin", (S, H, W))
    ok &= rep("conf", t, c, 0.100 * scale)
elif arch == "pi3x":
    # same output contract as pi3 (pose16 c2w row-major, local_points metric,
    # conf logits, points metric) + metric scale.
    # THRESHOLD CALIBRATION: pi3x's lp med_rel floor is ~0.015 for BOTH f16
    # and f32 (flash_attn_ext keeps K/V in f16 regardless of weight dtype;
    # the retrained pi3x decoder amplifies that noise ~9x through 36 blocks),
    # so the base threshold is pi3x-specific (pi3's floor is 0.0036).
    td = REF or "/tmp/pi3x_smoke/torch"     # prefix-style: <td>.pose16.bin ...
    t = load(f"{td}.pose16.bin", (S, 16))
    c = load(f"{prefix}.pose.bin", (S, 16))
    ok &= rep("pose16", t, c, 0.010 * scale, 0.030 * scale)
    t = load(f"{td}.local_points.bin", (S, H, W, 3))
    c = load(f"{prefix}.local_points.bin", (S, H, W, 3))
    ok &= rep("local_points", t, c, 0.020 * scale)
    t = load(f"{td}.conf.bin", (S, H, W))
    c = load(f"{prefix}.conf.bin", (S, H, W))
    ok &= rep("conf", t, c, 0.100 * scale)
    ts = float(load(f"{td}.scale.bin")[0])
    cs = float(load(f"{prefix}.scale.bin")[0])
    srel = abs(ts - cs) / ts
    print(f"  scale            torch {ts:.6f}  cpp {cs:.6f}  rel {srel:.5f}")
    ok &= srel < 0.010 * scale
elif arch == "mapanything":
    td = REF or "/tmp/ma_spy"               # dir-style: <td>/out_rays{v}.bin ...
    t = load(f"{td}/pose_raw.bin", (S, 7))
    c = load(f"{prefix}.pose_raw.bin", (S, 7))
    ok &= rep("pose_raw", t, c, 0.010 * scale, 0.050 * scale)
    t = np.stack([load(f"{td}/out_rays{v}.bin", (H, W, 3)) for v in range(S)])
    c = load(f"{prefix}.local_points.bin", (S, H, W, 3))
    ok &= rep("rays", t, c, 0.010 * scale)
    t = np.stack([load(f"{td}/out_depth{v}.bin", (H, W)) for v in range(S)])
    c = load(f"{prefix}.depth.bin", (S, H, W))
    ok &= rep("depth", t, c, 0.010 * scale)
    t = np.stack([load(f"{td}/out_conf{v}.bin", (H, W)) for v in range(S)])
    c = load(f"{prefix}.conf.bin", (S, H, W))
    ok &= rep("conf", t, c, 0.010 * scale)
    t = np.stack([load(f"{td}/out_pts3d{v}.bin", (H, W, 3)) for v in range(S)])
    c = load(f"{prefix}.points.bin", (S, H, W, 3))
    ok &= rep("pts3d", t, c, 0.020 * scale)
elif arch == "dust3r":
    # pair-wise M5: NO pose (the model has no pose head; global alignment is
    # an out-of-network optimizer). local_points s0 = head1 pts3d (view1
    # frame), s1 = head2 pts3d_in_other_view (view2's points, view1 frame);
    # conf = 1+exp; depth = pts3d z.  The spy dumps the two views as separate
    # (H,W,*) files, joined here in the s0/s1 order the C++ emits.
    # Resolution follows the actual dumps (the gate runs square res x res).
    td = REF or "/tmp/mapggml/ref/dust3r"    # dir-style: <td>/pts3d_view1.bin ...
    n = np.fromfile(f"{prefix}.local_points.bin", np.float32).size // 3
    H = W = int(round((n / S) ** 0.5))
    t = np.stack([load(f"{td}/pts3d_view1.bin", (H, W, 3)),
                  load(f"{td}/pts3d_view2_in_view1.bin", (H, W, 3))])
    c = load(f"{prefix}.local_points.bin", (S, H, W, 3))
    ok &= rep("local_points", t, c, 0.020 * scale)
    t = np.stack([load(f"{td}/conf_view1.bin", (H, W, 1))[..., 0],
                  load(f"{td}/conf_view2.bin", (H, W, 1))[..., 0]])
    c = load(f"{prefix}.conf.bin", (S, H, W))
    ok &= rep("conf", t, c, 0.100 * scale)
elif arch == "vggt":
    td = REF or "/tmp/vggt1b_torch"         # prefix-style: <td>.pose.bin ...
    t = load(f"{td}.pose.bin", (S, 9))
    c = load(f"{prefix}.pose.bin", (S, 9))
    ok &= rep("pose_enc", t, c, 0.005 * scale, 0.030 * scale)
    t = load(f"{td}.depth.bin", (S, H, W))
    c = load(f"{prefix}.depth.bin", (S, H, W))
    ok &= rep("depth", t, c, 0.005 * scale)
    t = load(f"{td}.depth_conf.bin", (S, H, W))
    c = load(f"{prefix}.depth_conf.bin", (S, H, W))
    ok &= rep("depth_conf", t, c, 0.030 * scale)
    # the official point head (view0-frame world pointmap): dump_torch_vggt
    # has always emitted .points.bin/.points_conf.bin, but the C++ CLI only
    # started writing them in 2026-09-24 — compare when both sides exist
    import os
    if os.path.exists(f"{td}.points.bin") and \
            os.path.exists(f"{prefix}.points.bin"):
        t = load(f"{td}.points.bin", (S, H, W, 3))
        c = load(f"{prefix}.points.bin", (S, H, W, 3))
        ok &= rep("points", t, c, 0.030 * scale)
        t = load(f"{td}.points_conf.bin", (S, H, W))
        c = load(f"{prefix}.points_conf.bin", (S, H, W))
        ok &= rep("points_conf", t, c, 0.100 * scale)
else:
    raise SystemExit(f"unknown arch {arch}")

print("  GATE: " + ("PASS" if ok else "FAIL") + f"  [{arch} {q}]")
sys.exit(0 if ok else 1)
