#!/usr/bin/env bash
# One-click parity gate for the cpp_ggml VGGT-Omega port.
#
# Steps:
#   1. (optional) regenerate the PyTorch reference npz + stage dumps
#   2. run the C++ CLI on the same frames
#   3. compare final outputs against the reference
#   4. optional stage-dump comparison (--dump-dir must match in 1+2)
#
# Usage:
#   scripts/e2e_gate.sh [--ckpt models/pytorch/vggt_omega_1b_512.pt] \
#                       [--gguf models/gguf/vggt-omega-1b-512-f16.gguf] \
#                       [--quant f16] [--frames /tmp/vggt_smoke/frames.bin] \
#                       [--H 512 --W 512 --S 2] [--stages /tmp/vggt_stages] \
#                       [--build-dir build-cpu] [--skip-torch]
#
# --ckpt implies the matching gguf under models/gguf/ unless --gguf is given
# explicitly (vggt_omega_1b_416_reproduce.pt -> vggt-omega-1b-416-reproduce).
set -euo pipefail

cd "$(dirname "$0")/.."   # cpp_ggml root

CKPT="models/pytorch/vggt_omega_1b_512.pt"
GGUF=""
QUANT="f16"
FRAMES="/tmp/vggt_smoke/frames.bin"
H=512; W=512; S=2
STAGES="/tmp/vggt_stages"
BUILD_DIR="build-cpu"
SKIP_TORCH=0
DUMP_DIR="${DUMP_DIR:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --gguf) GGUF="$2"; shift 2;;
    --quant) QUANT="$2"; shift 2;;
    --frames) FRAMES="$2"; shift 2;;
    --H) H="$2"; shift 2;;
    --W) W="$2"; shift 2;;
    --S) S="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --dump-dir) DUMP_DIR="$2"; shift 2;;
    --skip-torch) SKIP_TORCH=1; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

if [[ -z "$GGUF" ]]; then
  STEM="$(basename "$CKPT" .pt)"            # e.g. vggt_omega_1b_512
  STEM="vggt-omega-${STEM#vggt_omega_}"      # -> vggt-omega-1b_512
  STEM="${STEM//_/-}"                        # -> vggt-omega-1b-512
  GGUF="models/gguf/${STEM}-${QUANT}.gguf"
fi
MODEL="$GGUF"
OUT="/tmp/vggt_gate/out_${QUANT}"
mkdir -p "$STAGES" /tmp/vggt_gate

echo "== [1/3] build ($BUILD_DIR) =="
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -1

echo "== [2/3] torch reference =="
if [[ $SKIP_TORCH -eq 0 ]]; then
  python3 scripts/dump_torch_stages.py "$CKPT" "$FRAMES" "$H" "$W" "$S" "$STAGES"
else
  echo "  skipped (--skip-torch); expecting existing $STAGES/ref.npz"
fi
test -f "$STAGES/ref.npz" || { echo "missing $STAGES/ref.npz"; exit 1; }

echo "== [3/3] C++ inference ($MODEL) =="
CLI="$BUILD_DIR/bin/vggt-cli"
if [[ -n "$DUMP_DIR" ]]; then
  # the stage-dump directory travels as an explicit CLI flag
  # (RuntimeOptions::dump_dir) — never via the environment
  echo "  stage dumps enabled -> $DUMP_DIR (--dump-dir)"
  "$CLI" \
    --model "$MODEL" --bin "$FRAMES" --H "$H" --W "$W" --S "$S" \
    --out-prefix "$OUT" --dump-dir "$DUMP_DIR" | tail -3
else
  "$CLI" \
    --model "$MODEL" --bin "$FRAMES" --H "$H" --W "$W" --S "$S" \
    --out-prefix "$OUT" | tail -3
fi

echo "== parity report =="
python3 scripts/compare_parity.py "$OUT" "$STAGES/ref.npz"

echo "== gate =="
python3 - "$OUT" "$STAGES/ref.npz" <<'EOF'
import sys
import numpy as np
out, ref_npz = sys.argv[1], sys.argv[2]
ref = np.load(ref_npz)
pose = np.fromfile(f"{out}.pose.bin", dtype=np.float32).reshape(ref["pose_enc"].shape)
depth = np.fromfile(f"{out}.depth.bin", dtype=np.float32).reshape(ref["depth"].shape)
r = ref["depth"]
d = np.abs(depth - r)
rel = d / np.maximum(np.abs(r), 0.05)
pose_err = np.abs(pose - ref["pose_enc"]).max()
med_rel = float(np.median(rel))
print(f"pose_enc max_abs = {pose_err:.4f}   (gate < 0.005)")
print(f"depth median_rel = {med_rel:.4f}   (gate < 0.005)")
ok = pose_err < 0.005 and med_rel < 0.005
print("GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
