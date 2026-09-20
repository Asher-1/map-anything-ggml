#!/usr/bin/env bash
# =============================================================================
# run_mapggml.sh — multi-model pure C++ (ggml) inference, one-click launcher
#   supported: vggt-omega (512/416/text), vggt-1b, pi3, pi3x, mapanything,
#              dust3r
#
# MODEL SELECTION GUIDE (accuracy measured on the official MapAnything
# ETH3D protocol, 130 sets, seed 777 — see
# cpp_ggml/FEATURE_PARITY_AUDIT.md §6 for the full table):
#   vggt-omega  DEFAULT (512). Best accuracy across the board: pointmaps
#               0.0302, z_depth 0.0208, ATE 0.0065, rot 0.56°, AUC5 79.2
#               (official reference: 79.53). Also the best-quantization
#               -robust N-view model.
#   mapanything Pick this when you need METRIC SCALE and the best
#               camera poses: metric_scale_abs_rel 0.162 (far ahead of the
#               pack), ATE 0.0124, rot 0.92°, plus the fastest large model
#               (CUDA q8_0/q5_K 2.1-2.2x vs official torch).
#   dust3r      Pick this for RESOURCE-CONSTRAINED devices or maximum
#               THROUGHPUT: pair-wise (S=2) only, but CUDA 1.71-1.75x vs
#               official torch, smallest quantization degradation of the
#               whole family, and the lowest CPU latency.
#   pi3x        Balanced alternative to vggt-omega with a published paper
#               protocol; pi3/vggt-1b are kept for lineage/comparison.
#
# Subcommands:
#   demo    download -> convert -> build -> infer, end to end (newcomers:
#           just run ./run_mapggml.sh demo [--models pi3x])
#   setup   download the official ckpt(s) and convert to GGUF (= download + convert)
#   download  download the PyTorch checkpoint(s) only (HF; token only needed
#             for the gated facebook/VGGT-Omega repo)
#   convert   convert ckpt -> gguf only (default f16)
#   build     build the C++ runtime (auto-detect CUDA/Vulkan/CPU backend)
#   infer     run inference on images (--images required, or demo generates
#             synthetic test images)
#   gate      end-to-end parity gate (C++ vs PyTorch reference)
#   bench     end-to-end latency benchmark (P50/P95 over repeated runs)
#   help      help
#
# Shared parameters (all subcommands; recommended defaults):
#   --backend auto|cpu|cuda|vulkan|metal   backend selection (default auto)
#   --build-dir DIR                        build dir (default cpp_ggml/build-<backend>)
#   --jobs N                               parallel build jobs (default nproc)
#   --models LIST                          model aliases, space/comma separated:
#                                          512 416 text        (vggt-omega variants)
#                                          vggt-1b pi3 pi3x mapanything dust3r
#                                          or "all" (= every model above)
#                                          (default 512 = vggt-omega, the
#                                          most accurate model)
#   --quants LIST                          quants: f16,q8_0,q5_K,f32 or all
#                                          (default f16; all = f32 f16 q8_0 q5_K)
#   --dry-run                              print the commands without running
#
# infer/demo extra parameters:
#   --images a.jpg b.jpg                   input images (demo generates 2
#                                          synthetic test images when absent)
#   --views N                              view count S (default = image count)
#   --image-size N                         inference resolution (default follows
#                                          the model: 512/416/256/518)
#   --resize-mode balanced|max_size        official preprocessing mode (default balanced)
#   --out-prefix PATH                      output prefix (default /tmp/mapggml/out)
#   --quant Q                              quantization for inference (default f16)
#   --repeats N / --warmup N               in-process repeated inference (activates
#                                          CUDA graph replay; the last run's
#                                          outputs are written)
#
# gate extra: --H/--W/--S (default follows the model resolution), --skip-torch
#             (reuse an existing reference; without it the torch reference is
#             generated on demand), --regen-ref (force regeneration)
# bench extra: --repeats N (default 5), --warmup N (default 10, steady-state
#              protocol), --H/--W/--S
# build extra: --debug-dump (compile in the parity intermediate-dump hooks)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP="$REPO_ROOT/cpp_ggml"

# ----------------------------- defaults (recommended) ------------------------
CMD="${1:-demo}"; shift || true
BACKEND="auto"
BUILD_DIR=""
JOBS="$(nproc 2>/dev/null || echo 4)"
MODELS="512"
QUANTS="f16"
IMAGES=()
S=0
IMAGE_SIZE=""
OUT_PREFIX="/tmp/mapggml/out"
REPEATS=5
WARMUP=10
DEBUG_DUMP=OFF
SKIP_TORCH=0
REGEN_REF=0
DRY_RUN=0
# Optional extra PYTHONPATH for the torch-reference steps (gate); honored but
# never required — the spy scripts self-locate their model sources.
MAPGGML_PYTHONPATH="${MAPGGML_PYTHONPATH:-}"

usage() { sed -n '3,52p' "$0" | sed 's/^# \{0,1\}//'; }

# ----------------------------- argument parsing ------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --jobs|-j) JOBS="$2"; shift 2;;
    --models|--model) MODELS="$2"; shift 2;;
    --quants|--quant) QUANTS="$2"; shift 2;;
    --images) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do IMAGES+=("$1"); shift; done;;
    --views|--S) S="$2"; shift 2;;
    --H) H="$2"; shift 2;;
    --W) W="$2"; shift 2;;
    --image-size) IMAGE_SIZE="$2"; shift 2;;
    --resize-mode) RESIZE_MODE="$2"; shift 2;;
    --out-prefix) OUT_PREFIX="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --skip-torch) SKIP_TORCH=1; shift;;
    --regen-ref) REGEN_REF=1; shift;;
    --debug-dump) DEBUG_DUMP=ON; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1 (see --help)"; exit 1;;
  esac
done

# ----------------------------- helpers ---------------------------------------
run() {  # --dry-run support
  echo "+ $*"
  [[ $DRY_RUN -eq 1 ]] || "$@"
}

die() { echo "[run_mapggml] ERROR: $*" >&2; exit 1; }
info() { echo "[run_mapggml] $*"; }

# ----------------------------- model registry --------------------------------
# The SINGLE source of truth: alias -> "family ckpt_rel converter gguf_stem res"
#   family  = C++ general.architecture + output-contract group
#             (vggt_omega/vggt: pose_enc(S,9)+depth+depth_conf;
#              pi3/pi3x/mapanything: pose(S,16 c2w row-major)+local_points
#              +conf+points(+scale))
#   ckpt_rel= path under cpp_ggml/models/pytorch/ (what download fetches)
#   res     = default inference resolution (also the parity-frame resolution)
ALL_MODELS="512 416 text vggt-1b pi3 pi3x mapanything dust3r"
model_info() {
  case "$1" in
    512|omega512)        echo "vggt_omega vggt_omega_1b_512.pt convert_vggt_omega_to_gguf.py vggt-omega-1b-512 512";;
    416|omega416)        echo "vggt_omega vggt_omega_1b_416_reproduce.pt convert_vggt_omega_to_gguf.py vggt-omega-1b-416-reproduce 416";;
    text|omegatext)      echo "vggt_omega vggt_omega_1b_256_text.pt convert_vggt_omega_to_gguf.py vggt-omega-1b-256-text 256";;
    vggt|vggt1b|vggt-1b) echo "vggt vggt1b/model.pt convert_vggt_to_gguf.py vggt-1b 518";;
    pi3)                 echo "pi3 pi3/model.safetensors convert_pi3_to_gguf.py pi3 518";;
    pi3x)                echo "pi3x pi3x/model.safetensors convert_pi3x_to_gguf.py pi3x 518";;
    ma|mapanything)      echo "mapanything mapanything/model.safetensors convert_mapanything_to_gguf.py mapanything 518";;
    dust3r)              echo "dust3r dust3r/model.safetensors convert_dust3r_to_gguf.py dust3r 512";;
    *) return 1;;
  esac
}
model_field() {  # $1=alias  $2=field: 1=family 2=ckpt_rel 3=converter 4=gguf_stem 5=res
  model_info "$1" | awk -v f="$2" '{print $f}'
}
model_ckpt() { model_field "$1" 2; }
model_gguf() { model_field "$1" 4; }
model_res()  { model_field "$1" 5; }

expand_models() {  # "all" -> every registered alias
  if [[ "$MODELS" == "all" ]]; then echo "$ALL_MODELS"; else echo "$MODELS" | tr ',+' '  '; fi
}

expand_quants() {  # "all" -> all quants (f32 heaviest, kept last)
  if [[ "$QUANTS" == "all" ]]; then echo "f16 q8_0 q5_K f32"; else echo "$QUANTS" | tr ',+' '  '; fi
}

detect_backend() {
  if [[ "$BACKEND" != "auto" ]]; then echo "$BACKEND"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    # GPU present AND (nvcc available or an existing cuda build) -> cuda;
    # otherwise fall through to vulkan/cpu
    if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]] || [[ -d "$CPP/build-cuda" ]]; then
      echo cuda; return
    fi
  fi
  if command -v vulkaninfo >/dev/null 2>&1 || [[ -n "${VULKAN_SDK:-}" ]] || ls -d "$HOME"/VulkanSDK/*/x86_64 >/dev/null 2>&1; then
    echo vulkan; return
  fi
  echo cpu
}

resolve_backend_and_dir() {
  BACKEND="$(detect_backend)"
  [[ -z "$BUILD_DIR" ]] && BUILD_DIR="$CPP/build-$BACKEND"
  info "backend=$BACKEND  build-dir=$BUILD_DIR  jobs=$JOBS"

  # Vulkan SDK injection (the ggml Vulkan build needs glslc)
  if [[ "$BACKEND" == "vulkan" ]]; then
    if [[ -z "${VULKAN_SDK:-}" ]]; then
      local sdk; sdk="$(ls -d "$HOME"/VulkanSDK/*/x86_64 2>/dev/null | sort -V | tail -1 || true)"
      [[ -n "$sdk" ]] && export VULKAN_SDK="$sdk"
    fi
    [[ -n "${VULKAN_SDK:-}" && -d "$VULKAN_SDK/bin" ]] && export PATH="$VULKAN_SDK/bin:$PATH"
    command -v glslc >/dev/null 2>&1 || die "glslc not found; install the Vulkan SDK or use --backend cpu"
  fi
  # CUDA compiler injection
  if [[ "$BACKEND" == "cuda" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
      if [[ -x /usr/local/cuda/bin/nvcc ]]; then export PATH="/usr/local/cuda/bin:$PATH"; fi
    fi
    command -v nvcc >/dev/null 2>&1 || die "nvcc not found; install the CUDA toolkit or use --backend cpu/vulkan"
  fi
}

check_python_deps() {
  command -v python3 >/dev/null 2>&1 || die "python3 is required"
  for mod in torch gguf numpy; do
    python3 -c "import $mod" 2>/dev/null || die "missing python module: $mod (pip install $mod)"
  done
  if [[ "${1:-}" == "download" ]]; then
    python3 -c "import huggingface_hub" 2>/dev/null || die "missing huggingface_hub (pip install huggingface_hub)"
  fi
  # mapanything's torch side needs the uniception encoder library (a pip
  # dependency of the mapanything package, pyproject.toml pin 0.1.7)
  if [[ "${1:-}" == "mapanything-gate" ]]; then
    python3 -c "import uniception" 2>/dev/null \
      || die "missing python module: uniception (pip install uniception==0.1.7 — mapanything's encoder library)"
  fi
}

check_submodules() {
  # Fresh clones leave submodules empty; initialize on demand so a
  # newcomer's first run just works.  Optional arg: a single submodule
  # path suffix (e.g. vggt-src) to initialize just that one, checked by
  # its key source file (a leftover empty dir must not count as present).
  local need="${1:-}" missing=0
  if [[ -z "$need" ]]; then
    [[ -f "$CPP/third_party/ggml/src/ggml.c" \
       || -f "$CPP/third_party/ggml/src/ggml.cpp" ]] || missing=1
  else
    case "$need" in
      vggt-src) [[ -f "$CPP/third_party/vggt-src/vggt/models/vggt.py" ]] || missing=1 ;;
      *) [[ -n "$(ls -A "$CPP/third_party/$need" 2>/dev/null)" ]] || missing=1 ;;
    esac
  fi
  [[ $missing -eq 0 ]] && return 0
  local args=(submodule update --init --recursive --depth 1)
  [[ -n "$need" ]] && args+=("cpp_ggml/third_party/$need")
  info "initializing git submodule ${need:+($need)}..."
  run git -C "$REPO_ROOT" "${args[@]}"
}

cmd_download() {
  check_python_deps download
  local models; models="$(expand_models)"
  local only=()
  for m in $models; do only+=("$(model_ckpt "$m")"); done
  mkdir -p "$CPP/models/pytorch"
  if run python3 "$CPP/scripts/download_pytorch_ckpts.py" \
      --dir "$CPP/models/pytorch" --only "${only[@]}"; then
    return 0
  fi
  # Robustness: on direct HuggingFace failure retry via the hf-mirror.com
  # mirror (HF token auth is preserved).
  if [[ $DRY_RUN -eq 0 ]]; then
    info "direct HuggingFace download failed; retrying via the hf-mirror.com mirror..."
    HF_ENDPOINT=https://hf-mirror.com \
      python3 "$CPP/scripts/download_pytorch_ckpts.py" \
      --dir "$CPP/models/pytorch" --only "${only[@]}"
  fi
}

cmd_convert() {
  check_python_deps
  local models; models="$(expand_models)"
  local quants; quants="$(expand_quants)"
  # K-quants need libggml-cpu.so (ctypes-based quantization)
  if ! ls "$CPP"/build-*/lib/libggml-cpu.so >/dev/null 2>&1; then
    info "no built libggml-cpu.so found; building the CPU backend first (K-quant conversion depends on it)"
    cmd_build_inner cpu "$CPP/build-cpu" OFF
  fi
  for m in $models; do
    local ckpt gguf_stem converter
    ckpt="$CPP/models/pytorch/$(model_ckpt "$m")"
    gguf_stem="$(model_gguf "$m")"
    converter="$(model_field "$m" 3)"
    [[ -f "$ckpt" ]] || die "missing $ckpt; run: $0 download --models $m"
    for q in $quants; do
      local out="$CPP/models/gguf/${gguf_stem}-${q}.gguf"
      if [[ -f "$out" && $(stat -c%s "$out") -gt 1000000 ]]; then
        info "already exists, skipping: $out"
        continue
      fi
      mkdir -p "$CPP/models/gguf"
      run python3 "$CPP/scripts/$converter" \
          "$ckpt" "$out" --outtype "$q"
    done
  done
}

cmd_build_inner() {  # $1=backend $2=build-dir $3=debug_dump
  check_submodules
  local cfg_args=(-S "$CPP" -B "$2" -DCMAKE_BUILD_TYPE=Release -DMAPGGML_ENABLE_DUMP="$3")
  case "$1" in
    cuda)   cfg_args+=(-DMAP_GGML_CUDA=ON);;
    vulkan) cfg_args+=(-DMAP_GGML_VULKAN=ON);;
    metal)  cfg_args+=(-DMAP_GGML_METAL=ON);;
    cpu)    :;;
    *) die "unknown backend: $1";;
  esac
  echo "+ cmake ${cfg_args[*]}"
  [[ $DRY_RUN -eq 1 ]] || cmake "${cfg_args[@]}"
  echo "+ cmake --build $2 -j$JOBS"
  [[ $DRY_RUN -eq 1 ]] || cmake --build "$2" -j"$JOBS"
  [[ $DRY_RUN -eq 1 ]] || [[ -x "$2/bin/vggt-cli" ]] || die "build artifact missing: $2/bin/vggt-cli"
}

cmd_build() {
  resolve_backend_and_dir
  if cmd_build_inner "$BACKEND" "$BUILD_DIR" "$DEBUG_DUMP"; then
    info "build OK: $BUILD_DIR/bin/vggt-cli"
    return 0
  fi
  # Robustness: fall back to the CPU backend when a GPU build fails.
  if [[ "$BACKEND" != "cpu" ]]; then
    info "$BACKEND build failed; falling back to the CPU backend..."
    BACKEND=cpu; BUILD_DIR="$CPP/build-cpu"
    cmd_build_inner cpu "$BUILD_DIR" "$DEBUG_DUMP"
    info "CPU build OK: $BUILD_DIR/bin/vggt-cli"
  fi
}

pick_cli() {  # pick an available vggt-cli (prefer the current backend)
  local be dir
  be="$(detect_backend)"
  for dir in "$CPP/build-$be" "$CPP"/build-*/; do
    [[ -x "$dir/bin/vggt-cli" ]] && { echo "$dir/bin/vggt-cli"; return 0; }
  done
  return 1
}

gen_frames() {  # $1=out path $2=H $3=W $4=S: deterministic random smoke frames (parity)
  local frames="$1" h="$2" w="$3" s="$4"
  mkdir -p "$(dirname "$frames")"
  python3 - "$frames" "$h" "$w" "$s" <<'EOF'
import sys
import numpy as np
frames, h, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(42)
rng.random((s, 3, h, w), dtype=np.float32).astype(np.float32).tofile(frames)
EOF
}

gen_test_images() {  # generate 2 synthetic test images (demo fallback)
  local dir="/tmp/mapggml/test_images"
  mkdir -p "$dir"
  python3 - "$dir" <<'EOF'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
d = Path(sys.argv[1])
rng = np.random.default_rng(0)
for i, (w0, h0) in enumerate([(640, 480), (500, 620)]):
    yy, xx = np.mgrid[0:h0, 0:w0]
    img = np.stack([
        (xx / w0 * 255),                      # red: horizontal gradient
        (yy / h0 * 255),                      # green: vertical gradient
        (np.sin(xx / 24.0) * np.sin(yy / 18.0) * 60 + 128),  # blue texture
    ], axis=-1).astype(np.uint8)
    img = np.clip(img + rng.integers(-8, 8, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(d / f"test_{i}.jpg", quality=95)
print(d)
EOF
  IMAGES=("$dir/test_0.jpg" "$dir/test_1.jpg")
}

cmd_infer() {
  local cli; cli="$(pick_cli)" || die "vggt-cli not found; run: $0 build"
  local models; models="$(expand_models)"
  local quants; quants="$(expand_quants)"
  local m0 q0
  m0="$(echo "$models" | awk '{print $1}')"
  q0="$(echo "$quants" | awk '{print $1}')"
  local gguf="$CPP/models/gguf/$(model_gguf "$m0")-${q0}.gguf"
  [[ -f "$gguf" ]] || die "missing $gguf; run: $0 setup"
  [[ ${#IMAGES[@]} -gt 0 ]] || { info "no --images given; generating synthetic test images"; gen_test_images; }
  local isize; isize="${IMAGE_SIZE:-$(model_res "$m0")}"
  local s="${S:-0}"
  [[ "$s" -gt 0 ]] || s=${#IMAGES[@]}   # default S = image count
  local args=(--model "$gguf" --image-size "$isize" --out-prefix "$OUT_PREFIX" --timing
              --warmup "$WARMUP" --repeats "$REPEATS")
  [[ -n "${RESIZE_MODE:-}" ]] && args+=(--resize-mode "$RESIZE_MODE")
  local i
  for ((i=0; i<s; i++)); do args+=(--images "${IMAGES[$((i % ${#IMAGES[@]}))]}"); done
  info "inference: $cli ${args[*]}"
  run "$cli" "${args[@]}"
  local family; family="$(model_field "$m0" 1)"
  case "$family" in
    vggt_omega|vggt)
      info "outputs: ${OUT_PREFIX}.pose.bin / .depth.bin / .depth_conf.bin / .meta.json";;
    dust3r)
      info "outputs: ${OUT_PREFIX}.local_points.bin / .conf.bin / .depth.bin / .points.bin / .meta.json (NO pose: dust3r has no pose head)";;
    *)
      info "outputs: ${OUT_PREFIX}.pose.bin (c2w 4x4) / .local_points.bin / .depth.bin / .conf.bin / .points.bin / .meta.json"
      [[ -f "${OUT_PREFIX}.scale.bin" ]] && info "          + ${OUT_PREFIX}.scale.bin (metric_scaling_factor)";;
  esac
  # Package into the official demo's predictions.npz contract (best effort;
  # the packer auto-detects the vggt vs pi3-family contract from pose.bin)
  run python3 "$CPP/scripts/make_predictions_npz.py" --prefix "$OUT_PREFIX" \
    && info "official-contract npz: ${OUT_PREFIX}.predictions.npz" || true
  # Depth visualization (optional; skipped when matplotlib is missing)
  python3 "$CPP/scripts/viz_outputs.py" --prefix "$OUT_PREFIX" 2>/dev/null \
    && info "depth visualization: ${OUT_PREFIX}.depth_vis.png" || true
}

gen_torch_ref() {  # $1=family $2=ref $3=frames $4=res $5=S
  # Generate the torch reference dump for a non-omega family.  Every spy
  # script self-locates its model sources (third_party/pi3-src, repo root,
  # third_party/vggt-src); torch itself must be importable (checked earlier).
  local family="$1" ref="$2" frames="$3" res="$4" s="$5"
  local pypath="${MAPGGML_PYTHONPATH:+$MAPGGML_PYTHONPATH:}"
  mkdir -p "$(dirname "$ref")"
  case "$family" in
    pi3)
      run env PYTHONPATH="${pypath}." python3 "$CPP/scripts/spy_torch_pi3.py" \
          "$frames" "$res" "$res" "$s" "$ref";;
    pi3x)
      run env PYTHONPATH="${pypath}." python3 "$CPP/scripts/spy_torch_pi3x.py" \
          "$frames" "$res" "$res" "$s" "$ref";;
    mapanything)
      run env PYTHONPATH="${pypath}." python3 "$CPP/scripts/spy_torch_mapanything.py" \
          "$frames" "$res" "$res" "$s" "$ref";;
    dust3r)
      run env PYTHONPATH="${pypath}." python3 "$CPP/scripts/spy_torch_dust3r.py" \
          "$frames" "$res" "$res" "$s" "$ref";;
    vggt)
      # official sources come from the vggt-src git submodule (same mechanism
      # as vggt-omega-src / pi3-src; see .gitmodules)
      check_submodules vggt-src
      [[ -f "$CPP/third_party/vggt-src/vggt/models/vggt.py" ]] \
        || die "vggt-src submodule not available. Fix: git submodule update --init --depth 1 cpp_ggml/third_party/vggt-src — or point VGGT_SRC at an existing checkout"
      run env PYTHONPATH="${pypath}." python3 "$CPP/scripts/dump_torch_vggt.py" \
          "$frames" "$res" "$res" "$s" "$ref";;
    *) die "no torch-reference generator for family: $family";;
  esac
}

cmd_gate() {
  check_python_deps
  resolve_backend_and_dir
  local models; models="$(expand_models)"
  local m0; m0="$(echo "$models" | awk '{print $1}')"
  local family; family="$(model_field "$m0" 1)"
  local res; res="$(model_res "$m0")"
  local h="$res"; [[ "${H:-0}" -gt 0 ]] && h="$H"
  local w="$res"; [[ "${W:-0}" -gt 0 ]] && w="$W"
  local s=2; [[ "${S:-0}" -gt 0 ]] && s="$S"
  local frames="/tmp/vggt_smoke/frames_${res}.bin"
  local want_bytes=$(( s * 3 * h * w * 4 ))
  # Regenerate the frames file when a stale one has a mismatching size
  if [[ -f "$frames" && $(stat -c%s "$frames") -ne "$want_bytes" ]]; then
    rm -f "$frames"
  fi
  # Prefer real-image frames (in-distribution, numerically stable); fall back
  # to random frames only when no image asset exists (pure noise is
  # out-of-distribution and the 416_reproduce ckpt's pose components may
  # exceed thresholds on it — a known numeric sensitivity).
  local teaser="$REPO_ROOT/assets/teaser.png"
  if [[ ! -f "$frames" && -f "$teaser" ]]; then
    info "generating parity frames from the real image: $teaser"
    python3 - "$teaser" "$frames" "$res" "$h" "$s" <<'EOF'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
tsr, frames, res, h, s = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
img = Image.open(tsr).convert("RGB")
w0, h0 = img.size
ar = h0 / w0
if ar < 0.5:
    cw = min(w0, max(1, round(h0 / 0.5))); img = img.crop(((w0 - cw) // 2, 0, (w0 - cw) // 2 + cw, h0))
elif ar > 2.0:
    ch = min(h0, max(1, round(w0 * 2.0))); img = img.crop((0, (h0 - ch) // 2, w0, (h0 - ch) // 2 + ch))
w0, h0 = img.size
img = img.resize((res, res), Image.BICUBIC)
x = np.asarray(img).astype(np.float32) / 255.0
x = np.stack([x] * s).transpose(0, 3, 1, 2)   # (S,3,h,w)
x.tofile(frames)
EOF
  elif [[ ! -f "$frames" ]]; then
    info "generating random smoke frames: $frames (${h}x${w}x${s})"
    gen_frames "$frames" "$h" "$w" "$s"
  fi
  if [[ "$family" == "vggt_omega" ]]; then
    # self-contained omega gate (generates its own torch reference)
    run bash "$CPP/scripts/e2e_gate.sh" \
        --ckpt "models/pytorch/$(model_ckpt "$m0")" \
        --quant "$(echo "$(expand_quants)" | awk '{print $1}')" \
        --build-dir "$BUILD_DIR" \
        --H "$h" --W "$w" --S "$s" \
        --frames "$frames" \
        $( [[ $SKIP_TORCH -eq 1 ]] && echo --skip-torch )
    return 0
  fi

  # Generic family gate: torch spy reference -> CLI -> cmp_gate_matrix.
  # Reference cache layout per family (dir-style vs prefix-style follows the
  # comparator's join convention):
  local ref
  case "$family" in
    pi3)         ref="/tmp/mapggml/ref/pi3";;          # <ref>/pose16.bin ...
    pi3x)        ref="/tmp/mapggml/ref/pi3x/torch";;   # <ref>.pose16.bin ...
    mapanything) ref="/tmp/mapggml/ref/mapanything";;  # <ref>/pose_raw.bin ...
    dust3r)      ref="/tmp/mapggml/ref/dust3r";;       # <ref>/pts3d_view1.bin ...
    vggt)        ref="/tmp/mapggml/ref/vggt-1b";;      # <ref>.pose.bin ...
    *) die "no gate path for family: $family";;
  esac
  local have_ref=0
  case "$family" in
    pi3)         [[ -f "$ref/pose16.bin" ]] && have_ref=1;;
    pi3x)        [[ -f "${ref}.pose16.bin" ]] && have_ref=1;;
    mapanything) [[ -f "$ref/pose_raw.bin" ]] && have_ref=1;;
    dust3r)      [[ -f "$ref/pts3d_view1.bin" ]] && have_ref=1;;
    vggt)        [[ -f "${ref}.pose.bin" ]] && have_ref=1;;
  esac
  if [[ $have_ref -eq 0 || $REGEN_REF -eq 1 ]]; then
    [[ $SKIP_TORCH -eq 1 ]] && die "--skip-torch requested but no torch reference at $ref (drop the flag to generate it)"
    [[ "$family" == "mapanything" ]] && check_python_deps mapanything-gate
    info "generating torch reference for $m0 ($family)..."
    gen_torch_ref "$family" "$ref" "$frames" "$res" "$s"
  fi
  local q0; q0="$(echo "$(expand_quants)" | awk '{print $1}')"
  local gguf="$CPP/models/gguf/$(model_gguf "$m0")-${q0}.gguf"
  [[ -f "$gguf" ]] || die "missing $gguf; run: $0 convert --models $m0"
  local cli; cli="$(pick_cli)" || die "vggt-cli not found; run: $0 build"
  local outp="/tmp/mapggml/gate_${m0}_${q0}"
  run "$cli" --model "$gguf" --bin "$frames" --H "$h" --W "$w" --S "$s" \
      --out-prefix "$outp"
  run python3 "$CPP/scripts/cmp_gate_matrix.py" "$family" "$outp" "$q0" "$ref"
}

cmd_bench() {
  check_python_deps
  resolve_backend_and_dir
  local cli; cli="$(pick_cli)" || die "vggt-cli not found; run: $0 build"
  local models; models="$(expand_models)"
  local quants; quants="$(expand_quants)"
  local m0 q0
  m0="$(echo "$models" | awk '{print $1}')"
  q0="$(echo "$quants" | awk '{print $1}')"
  local res; res="$(model_res "$m0")"
  local h="$res"; [[ "${H:-0}" -gt 0 ]] && h="$H"
  local w="$res"; [[ "${W:-0}" -gt 0 ]] && w="$W"
  local s=2; [[ "${S:-0}" -gt 0 ]] && s="$S"
  local frames="/tmp/vggt_smoke/frames_${h}x${w}x${s}.bin"
  local want_bytes=$(( s * 3 * h * w * 4 ))
  if [[ -f "$frames" && $(stat -c%s "$frames") -ne "$want_bytes" ]]; then
    rm -f "$frames"
  fi
  [[ -f "$frames" ]] || gen_frames "$frames" "$h" "$w" "$s"
  run python3 "$CPP/benchmarks/bench_latency.py" \
      --cli "$cli" --models-dir "$CPP/models/gguf" \
      --model "$(model_gguf "$m0")" --quants "$quants" \
      --frames "$frames" --H "$h" --W "$w" --S "$s" \
      --repeats "$REPEATS" --warmup "$WARMUP" \
      --out "$CPP/benchmarks/results/$(model_gguf "$m0")"
}

cmd_setup() { cmd_download; cmd_convert; }

cmd_demo() {
  info "== 1/3 download + convert models =="
  cmd_setup
  info "== 2/3 build the C++ runtime =="
  cmd_build
  info "== 3/3 inference =="
  cmd_infer
  info "All done! See $CPP/README.md for parity verification and the quantization matrix."
}

case "$CMD" in
  download) cmd_download;;
  convert)  cmd_convert;;
  build)    cmd_build;;
  infer)    cmd_infer;;
  gate)     cmd_gate;;
  bench)    cmd_bench;;
  setup)    cmd_setup;;
  demo)     cmd_demo;;
  help|-h|--help) usage;;
  *) die "unknown subcommand: $CMD (see --help)";;
esac
