#!/bin/bash
# Rerun the full ETH3D eval matrix (one md per model-version).
# Current matrix config: q4_K dropped, cloud (Chamfer-family) metrics on.
#
# Usage: bash scripts/rerun_eval_matrix.sh [DATA_ROOT]
#   DATA_ROOT defaults to <repo-root>/data/map-anything-benchmarking and
#   must contain eth3d/ (WAI scenes) and metadata/ (test scene lists).
#   ONLY=name1,name2 restricts to those versions (e.g. ONLY=512_f16,text_torch).
# Requires PYTHONPATH to expose CUDA torch + mapanything repo root, e.g.
#   PYTHONPATH=/tmp/torch_cuda_lib:<repo-root> bash scripts/rerun_eval_matrix.sh
set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${1:-$REPO_ROOT/data/map-anything-benchmarking}"
CLI=build-cuda/bin/vggt-cli

want() { [[ -z "${ONLY:-}" || ",$ONLY," == *",$1,"* ]]; }

run() { # name gguf source W H pt
  want "$1" || return 0
  echo "=== $1 ==="
  python3 scripts/eval_cpp_cli.py \
    --data-root "$DATA/eth3d" --metadata-dir "$DATA/metadata" \
    --cli "$CLI" --gguf "models/gguf/$2" --pred-source "$3" \
    --resolution "$4" "$5" --pt "$6" \
    --num-views 2 --max-sets 13 \
    --out "benchmarks/results/vggt-omega/eval_eth3d_$1.md"
}

run 512_torch   vggt-omega-1b-512-f16.gguf           torch 512 336 vggt-omega/vggt_omega_1b_512.pt
run 512_f16     vggt-omega-1b-512-f16.gguf           cpp   512 336 vggt-omega/vggt_omega_1b_512.pt
run 512_q8_0    vggt-omega-1b-512-q8_0.gguf          cpp   512 336 vggt-omega/vggt_omega_1b_512.pt
run 512_q6_K    vggt-omega-1b-512-q6_K.gguf          cpp   512 336 vggt-omega/vggt_omega_1b_512.pt
run 416_torch   vggt-omega-1b-416-reproduce-f16.gguf torch 416 272 vggt-omega/vggt_omega_1b_416_reproduce.pt
run 416_f16     vggt-omega-1b-416-reproduce-f16.gguf cpp   416 272 vggt-omega/vggt_omega_1b_416_reproduce.pt
run 416_q8_0    vggt-omega-1b-416-reproduce-q8_0.gguf cpp  416 272 vggt-omega/vggt_omega_1b_416_reproduce.pt
run 416_q6_K    vggt-omega-1b-416-reproduce-q6_K.gguf cpp  416 272 vggt-omega/vggt_omega_1b_416_reproduce.pt
run text_torch  vggt-omega-1b-256-text-f16.gguf      torch 256 176 vggt-omega/vggt_omega_1b_256_text.pt
run text_f16    vggt-omega-1b-256-text-f16.gguf      cpp   256 176 vggt-omega/vggt_omega_1b_256_text.pt
run text_q8_0   vggt-omega-1b-256-text-q8_0.gguf     cpp   256 176 vggt-omega/vggt_omega_1b_256_text.pt
run text_q6_K   vggt-omega-1b-256-text-q6_K.gguf     cpp   256 176 vggt-omega/vggt_omega_1b_256_text.pt
echo "ALL EVALS DONE"
