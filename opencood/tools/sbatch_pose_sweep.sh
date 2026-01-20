#!/usr/bin/env bash
#SBATCH --output=/home/qqxluca/runs/logs/slurm/%x-%j.out
set -euo pipefail

# Slurm entrypoint for pose-noise robustness sweeps (inference-only).
#
# Usage (submit via sbatch):
#   sbatch -J <name> \
#     HEAL/opencood/tools/sbatch_pose_sweep.sh \
#       --model-dir opencood/logs/<run_dir> \
#       --pose-correction none|v2xregpp_initfree|freealign_paper|freealign_repo \
#       --note _comm200_paper \
#       [--stage1-result opencood/logs/<stage1>/test/stage1_boxes.json]
#
# This script assumes repo layout: /home/qqxluca/v2xreg_private/HEAL.

MODEL_DIR=""
POSE_CORR="none"
STAGE1_RESULT=""
NOTE=""
COMM_RANGE="200"
NOISE_TARGET="non-ego"
POS_STD_LIST="0,1,2,3,4"
ROT_STD_LIST="0,1,2,3,4"
NUM_WORKERS="0"
LOG_INTERVAL="400"
SAVE_VIS_INTERVAL="100000000"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2;;
    --pose-correction) POSE_CORR="$2"; shift 2;;
    --stage1-result) STAGE1_RESULT="$2"; shift 2;;
    --note) NOTE="$2"; shift 2;;
    --comm-range) COMM_RANGE="$2"; shift 2;;
    --noise-target) NOISE_TARGET="$2"; shift 2;;
    --pos-std-list) POS_STD_LIST="$2"; shift 2;;
    --rot-std-list) ROT_STD_LIST="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --log-interval) LOG_INTERVAL="$2"; shift 2;;
    --save-vis-interval) SAVE_VIS_INTERVAL="$2"; shift 2;;
    --) shift; EXTRA_ARGS+=("$@"); break;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "${MODEL_DIR}" ]]; then
  echo "--model-dir is required" >&2
  exit 2
fi

cd /home/qqxluca/v2xreg_private/HEAL

# Stream Python stdout/stderr to Slurm logs for live monitoring.
export PYTHONUNBUFFERED=1

# Avoid micromamba locking inside Slurm jobs by calling the env Python directly.
HEAL_PYTHON="${HEAL_PYTHON:-$HOME/.micromamba/envs/heal/bin/python}"
if [[ ! -x "${HEAL_PYTHON}" ]]; then
  echo "heal python not found/executable at: ${HEAL_PYTHON}" >&2
  exit 2
fi

CMD=(
  "${HEAL_PYTHON}" opencood/tools/inference_w_noise.py
  --model_dir "${MODEL_DIR}"
  --fusion_method intermediate
  --pos-std-list "${POS_STD_LIST}"
  --rot-std-list "${ROT_STD_LIST}"
  --sweep-mode paired
  --noise-target "${NOISE_TARGET}"
  --comm-range-override "${COMM_RANGE}"
  --num-workers "${NUM_WORKERS}"
  --save_vis_interval "${SAVE_VIS_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --note "${NOTE}"
  --pose-correction "${POSE_CORR}"
)

if [[ "${POSE_CORR}" != "none" ]]; then
  if [[ -z "${STAGE1_RESULT}" ]]; then
    echo "--stage1-result is required for pose-correction=${POSE_CORR}" >&2
    exit 2
  fi
  CMD+=(--stage1-result "${STAGE1_RESULT}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[sbatch_pose_sweep] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[sbatch_pose_sweep] cmd: ${CMD[*]}"

"${CMD[@]}"
