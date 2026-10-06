#!/usr/bin/env bash
# wait_and_refresh.sh — poll until the GPU is EXCLUSIVE, then run the
# latency/chart refresh. Intended to be launched with nohup:
#   nohup bash benchmarks/wait_and_refresh.sh > wait_and_refresh.log 2>&1 &
# Exits after the refresh completes, or after MAX_WAIT_H of waiting.
set -u
cd "$(dirname "$0")/.."
MAX_WAIT_H=${MAX_WAIT_H:-24}
DEADLINE=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
POLL=${POLL:-60}

free_gpu() {
  local occ
  occ=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | tr -d ' ' | grep -c . || true)
  [ "$occ" -eq 0 ]
}

echo "[wait] started $(date '+%F %T'); polling every ${POLL}s for an idle GPU"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if free_gpu; then
    sleep 30  # confirmation gap: reject momentary idle windows
    if free_gpu; then
      echo "[wait] GPU exclusive at $(date '+%F %T') — starting refresh"
      bash benchmarks/refresh_latency_charts.sh
      rc=$?
      echo "[wait] refresh finished rc=$rc at $(date '+%F %T')"
      exit $rc
    fi
    echo "[wait] idle window closed again; keep polling"
  fi
  sleep "$POLL"
done
echo "[wait] gave up after ${MAX_WAIT_H}h at $(date '+%F %T')"
exit 124
