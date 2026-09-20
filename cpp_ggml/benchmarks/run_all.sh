#!/usr/bin/env bash
# One-click benchmark pipeline: build -> (torch reference) -> latency ->
# e2e accuracy -> charts. Backend selectable; run from cpp_ggml/ or anywhere.
#
# Usage:
#   benchmarks/run_all.sh [--backend cpu|cuda|vulkan] [--quants "f16 q8_0 q5_K q4_K"]
#                         [--model vggt-omega-1b-512] [--S 2] [--repeats 5]
#                         [--skip-torch-ref] [--skip-pytorch-baseline]
set -euo pipefail

CPP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CPP"

BACKEND="cpu"
QUANTS="f16 q8_0 q5_K q4_K"
MODEL="vggt-omega-1b-512"
S=2
REPEATS=5
SKIP_TORCH_REF=0
SKIP_PT_BASELINE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --quants) QUANTS="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --S) S="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --skip-torch-ref) SKIP_TORCH_REF=1; shift;;
    --skip-pytorch-baseline) SKIP_PT_BASELINE=1; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

BUILD_DIR="$CPP/build-$BACKEND"
CLI="$BUILD_DIR/bin/vggt-cli"
[[ -x "$CLI" ]] || { echo "missing $CLI — build first (run_mapggml.sh build --backend $BACKEND)"; exit 1; }

# per-checkpoint resolution: 512/416/256
RES=512
[[ "$MODEL" == *416* ]] && RES=416
[[ "$MODEL" == *256* ]] && RES=256

FRAMES="/tmp/vggt_smoke/frames_${RES}.bin"
STAGES="/tmp/vggt_stages_${RES}"
mkdir -p benchmarks/results
if [[ ! -f "$FRAMES" ]]; then
  python3 - "$FRAMES" "$RES" "$S" <<'EOF'
import sys
import numpy as np
frames, h, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[4])
rng = np.random.default_rng(42)
rng.random((s, 3, h, h), dtype=np.float32).astype(np.float32).tofile(frames)
EOF
fi

if [[ $SKIP_TORCH_REF -eq 0 && ! -f "$STAGES/ref.npz" ]]; then
  CKPT="models/pytorch/$(ls models/pytorch | grep -E "512|416|256" | grep -E "${RES}" | head -1)"
  echo "== torch reference ($CKPT) =="
  python3 scripts/dump_torch_stages.py "$CKPT" "$FRAMES" "$RES" "$RES" "$S" "$STAGES"
fi

echo "== latency ($BACKEND) =="
python3 benchmarks/bench_latency.py --cli "$CLI" --models-dir models/gguf \
  --model "$MODEL" --quants "$QUANTS" --frames "$FRAMES" \
  --H "$RES" --W "$RES" --S "$S" --repeats "$REPEATS" --warmup 2 \
  --out benchmarks/results

if [[ $SKIP_TORCH_REF -eq 0 ]]; then
  echo "== e2e accuracy =="
  python3 benchmarks/bench_e2e.py --cli "$CLI" --models-dir models/gguf \
    --model "$MODEL" --quants "$QUANTS" --ref "$STAGES/ref.npz" \
    --frames "$FRAMES" --H "$RES" --W "$RES" --S "$S" \
    --out benchmarks/results
fi

if [[ $SKIP_PT_BASELINE -eq 0 ]]; then
  echo "== PyTorch-CUDA baseline =="
  python3 benchmarks/bench_pytorch_baseline.py \
    --ckpt "models/pytorch/vggt_omega_1b_${RES}.pt" --frames "$FRAMES" \
    --H "$RES" --W "$RES" --S "$S" --repeats "$REPEATS" --warmup 2 \
    --out benchmarks/results || echo "baseline skipped (no CUDA?)"
fi

echo "== charts =="
python3 benchmarks/plot_charts.py --results benchmarks/results --out benchmarks/charts
echo "done. See benchmarks/charts/ and benchmarks/RESULTS.md"
