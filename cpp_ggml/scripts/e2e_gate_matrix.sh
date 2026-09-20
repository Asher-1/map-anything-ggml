#!/usr/bin/env bash
# (arch x quant x backend) parity gate matrix against fixed torch f32 refs.
#
# Usage:
#   scripts/e2e_gate_matrix.sh                # full matrix
#   scripts/e2e_gate_matrix.sh pi3 f16 q8_0   # arch/quant filter
set -uo pipefail
cd "$(dirname "$0")/.."

S=2   # every supported family gates at S=2 (dust3r is pair-wise)
OUT=/tmp/gate_matrix/out
mkdir -p /tmp/gate_matrix

declare -A GGUF=(
  [pi3_f16]=models/gguf/pi3-f16.gguf
  [pi3_f32]=models/gguf/pi3-f32.gguf
  [pi3_q8_0]=models/gguf/pi3-q8_0.gguf
  [pi3_q5_K]=models/gguf/pi3-q5_K.gguf
  [pi3x_f16]=models/gguf/pi3x-f16.gguf
  [pi3x_f32]=models/gguf/pi3x-f32.gguf
  [pi3x_q8_0]=models/gguf/pi3x-q8_0.gguf
  [pi3x_q5_K]=models/gguf/pi3x-q5_K.gguf
  [mapanything_f16]=models/gguf/mapanything-f16.gguf
  [mapanything_q8_0]=models/gguf/mapanything-q8_0.gguf
  [mapanything_q5_K]=models/gguf/mapanything-q5_K.gguf
  [vggt_f16]=models/gguf/vggt-1b-f16.gguf
  [vggt_f32]=models/gguf/vggt-1b-f32.gguf
  [vggt_q8_0]=models/gguf/vggt-1b-q8_0.gguf
  [vggt_q5_K]=models/gguf/vggt-1b-q5_K.gguf
  [dust3r_f16]=models/gguf/dust3r-f16.gguf
  [dust3r_f32]=models/gguf/dust3r-f32.gguf
  [dust3r_q8_0]=models/gguf/dust3r-q8_0.gguf
  [dust3r_q5_K]=models/gguf/dust3r-q5_K.gguf
)
# dust3r runs at 512x512 (patch 16, pair-wise); the N-view families at 518
# (patch 14). The cmp refs default to each arch's historical cache path.
declare -A RES=([dust3r]=512)
ARCHS=(${1:-pi3 pi3x mapanything vggt dust3r})
QUANTS=(${2:-f16 f32 q8_0 q5_K})
BUILDS=(build-cpu build-cuda build-vulkan)

fails=0; total=0
for arch in "${ARCHS[@]}"; do
  for q in "${QUANTS[@]}"; do
    gguf="${GGUF[${arch}_${q}]-}"
    [[ -z "$gguf" || ! -f "$gguf" ]] && continue
    for build in "${BUILDS[@]}"; do
      [[ -x "$build/bin/vggt-cli" ]] || continue
      total=$((total + 1))
      echo "== $arch $q $build =="
      res="${RES[$arch]:-518}"
      frames=/tmp/frames${res}.bin
      if ! "$build/bin/vggt-cli" --model "$gguf" --bin "$frames" \
           --H "$res" --W "$res" --S "$S" --out-prefix "$OUT" >/dev/null 2>&1; then
        echo "  CLI FAILED"
        fails=$((fails + 1)); continue
      fi
      if ! python3 scripts/cmp_gate_matrix.py "$arch" "$OUT" "$q"; then
        fails=$((fails + 1))
      fi
    done
  done
done
echo "=================================="
echo "MATRIX: $((total - fails))/$total PASS, $fails FAIL"
exit $((fails > 0))
