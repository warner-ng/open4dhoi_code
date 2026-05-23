#!/usr/bin/env bash
# ==============================================================================
# Step 3: Hand pose refinement using SAM-3D-Body
# Conda environment: mhr
#
# Usage:
#   bash step3_hand.sh <video_dir> [--retarget] [--smooth_cutoff 0.3]
# ==============================================================================
set -eo pipefail

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${PIPELINE_DIR}/config.sh"

VIDEO_DIR="${1:?Usage: $0 <video_dir> [extra_args...]}"
VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"
shift  # Remove video_dir from args, pass rest through

echo "============================================"
echo "Step 3: Hand Pose Refinement (SAM-3D-Body)"
echo "Session: ${VIDEO_DIR}"
echo "Conda env: ${ENV_MHR}"
echo "============================================"

eval "$(conda shell.bash hook)"
set +u
conda activate "${ENV_MHR}"
set -u

# Ensure conda runtime libraries are preferred over system libs.
# This avoids GLIBCXX mismatch errors (e.g. optree requiring GLIBCXX_3.4.31).
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo ""
echo "[1/1] Refining hand pose..."
if [ -f "${VIDEO_DIR}/motion/result_hand.pt" ]; then
    echo "  result_hand.pt already exists, skipping."
else
    CUDA_VISIBLE_DEVICES="${CUDA_HAND}" \
    python "${SCRIPT_DIR}/make_hand_sam3d.py" \
        --video_dir "${VIDEO_DIR}" \
        --checkpoint "${SAM3D_BODY_CHECKPOINT}" \
        --mhr_path "${SAM3D_BODY_MHR}" \
        --smplx_path "${SMPLX_MODEL}" \
        "$@"
    echo "  Hand pose refined."
fi

echo ""
echo "Step 3 complete."
