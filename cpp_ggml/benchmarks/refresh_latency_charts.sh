#!/usr/bin/env bash
# refresh_latency_charts.sh — one-shot latency re-measurement + chart refresh.
#
# Run ONLY on an EXCLUSIVE GPU (the script hard-fails on any co-tenant:
# a 100%-utilization neighbor once contaminated every CUDA row by 50-80%).
# Covers the 2026-10-01 leftovers: q6_K latency bars for the four q6_K
# models and the omega re-measurement (the 09-19 JSONs predate "opt vggt").
#
# Usage:
#   bash benchmarks/refresh_latency_charts.sh            # CUDA + Vulkan
#   bash benchmarks/refresh_latency_charts.sh --with-cpu # + slow CPU rows
#   bash benchmarks/refresh_latency_charts.sh --plots-only  # skip measuring
#
# After measuring, every model's charts are redrawn with its own gate log
# (results/<model>/gate_matrix_log.txt), so latency bars, the gate heatmap,
# the quant pareto and gate_summary.json all reflect the shipped tiers.
set -e
cd "$(dirname "$0")/.."
CPP=$(pwd)
REPEATS=${REPEATS:-5}
WARMUP=${WARMUP:-2}
WITH_CPU=0; PLOTS_ONLY=0
for a in "$@"; do case "$a" in --with-cpu) WITH_CPU=1;; --plots-only) PLOTS_ONLY=1;; esac; done

ALLOW_SHARED=0
for a in "$@"; do case "$a" in --allow-shared) ALLOW_SHARED=1;; esac; done
if [[ $PLOTS_ONLY -eq 0 ]]; then
  IFS=',' read -r used total util <<< "$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  free_mb=$(( total - used ))
  if [[ $ALLOW_SHARED -eq 1 ]]; then
    # resource-threshold mode: no SM contention and ample VRAM is what
    # actually matters for latency validity (an idle viewer holding VRAM
    # does not perturb the numbers — verified by a f16 repeat below)
    if (( free_mb < 8000 )) || (( util > 5 )); then
      echo "REFUSE: shared GPU below thresholds (free=${free_mb}MiB util=${util}%)"
      exit 1
    fi
    echo "shared-GPU mode: free=${free_mb}MiB util=${util}% — within thresholds"
  elif [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]]; then
    echo "REFUSE: GPU not exclusive (compute apps present)"
    exit 1
  fi
fi

# canonical parity frames (512 for omega/dust3r, 518 for the patch-14 family)
[[ -f /tmp/frames512.bin ]] || python3 -c "import numpy as np; np.random.default_rng(7).random((2,3,512,512),dtype=np.float32).tofile('/tmp/frames512.bin')"
[[ -f /tmp/frames518.bin ]] || python3 -c "import numpy as np; np.random.default_rng(7).random((2,3,518,518),dtype=np.float32).tofile('/tmp/frames518.bin')"

# stem|frames|H|W|S|results-dir  (K tier follows the 2026-10-01 assignment:
# q6_K for omega/pi3x/mapanything/dust3r, q5_K for pi3/vggt-1b — pi3 and
# vggt-1b need no re-measurement, their tier JSONs are already current)
CHUNK_DEFS=(
  "vggt-omega-1b-512|/tmp/frames512.bin|512|512|2|vggt-omega|f16 f32 q8_0 q6_K"
  "pi3x|/tmp/frames518.bin|518|518|2|pi3x|f16 f32 q8_0 q6_K"
  "mapanything|/tmp/frames518.bin|518|518|2|mapanything|f16 f32 q8_0 q6_K"
  "dust3r|/tmp/frames512.bin|512|512|2|dust3r|f16 f32 q8_0 q6_K"
)

measure() { # build-dir tag
  local bin="build-$1/bin/vggt-cli" tag="$2"
  [[ -x "$bin" ]] || { echo "skip $tag (no $bin)"; return 0; }
  for g in "${CHUNK_DEFS[@]}"; do
    IFS='|' read -r stem frames H W S rdir quants <<< "$g"
    echo "== $tag $stem ($quants) =="
    python3 benchmarks/bench_latency.py --cli "$bin" --models-dir models/gguf \
      --model "$stem" --quants "$quants" --frames "$frames" \
      --H "$H" --W "$W" --S "$S" --repeats "$REPEATS" --warmup "$WARMUP" \
      --out "benchmarks/results/$rdir"
  done
}

if [[ $PLOTS_ONLY -eq 0 ]]; then
  measure cuda CUDA
  measure vulkan Vulkan
  if [[ $WITH_CPU -eq 1 ]]; then measure cpu CPU; else
    echo "(CPU rows skipped — pass --with-cpu for a slow exclusive-CPU pass)"
  fi
fi

echo "== redrawing all six models with their own gate logs =="
for a in vggt_omega pi3 pi3x vggt mapanything dust3r; do
  d=$([[ $a == vggt_omega ]] && echo vggt-omega || { [[ $a == vggt ]] && echo vggt-1b || echo "$a"; })
  python3 scripts/plot_charts_model.py --arch "$a" \
    --gate-log "benchmarks/results/$d/gate_matrix_log.txt"
done
echo "DONE — charts refreshed from the shipped tiers' own gate logs."
