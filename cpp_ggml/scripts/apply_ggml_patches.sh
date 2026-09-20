#!/usr/bin/env bash
# apply_ggml_patches.sh
#
# Apply the in-tree ggml integration patches to third_party/ggml, in sorted
# order. Idempotent: a patch whose resulting files are already present (i.e.
# `git apply --check` fails with "already exists") is treated as applied and
# skipped. Zero patches is a valid configuration.
#
# Usage:
#   bash scripts/apply_ggml_patches.sh
#
# Exits 0 on success, non-zero on any failure. Designed to be called by CMake
# during configure but also runnable standalone for debugging.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_DIR="${PROJECT_ROOT}/third_party/ggml"
PATCH_DIR="${PROJECT_ROOT}/third_party/ggml-patches"

if [[ ! -d "${GGML_DIR}" ]]; then
    echo "error: ggml submodule not found at ${GGML_DIR}" >&2
    echo "       did you forget 'git submodule update --init --recursive'?" >&2
    exit 1
fi

if [[ ! -d "${PATCH_DIR}" ]]; then
    echo "no ggml patch directory (${PATCH_DIR}); nothing to apply"
    exit 0
fi

shopt -s nullglob
PATCHES=("${PATCH_DIR}"/*.patch)
shopt -u nullglob

if [[ ${#PATCHES[@]} -eq 0 ]]; then
    echo "no ggml patches found; nothing to apply"
    exit 0
fi

cd "${GGML_DIR}"

for PATCH in "${PATCHES[@]}"; do
    PATCH_NAME="$(basename "${PATCH}")"
    if git apply --check --reverse "${PATCH}" >/dev/null 2>&1; then
        echo "patch already applied (reverse-check ok): ${PATCH_NAME}"
        continue
    fi
    if git apply --3way "${PATCH}" >"/tmp/ggml-patch-${PATCH_NAME}.log" 2>&1; then
        echo "applied: ${PATCH_NAME}"
    else
        echo "error: failed to apply ${PATCH_NAME}; log follows:" >&2
        cat "/tmp/ggml-patch-${PATCH_NAME}.log" >&2
        exit 1
    fi
done

echo "ggml patches: ${#PATCHES[@]} processed"
exit 0
