#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "Usage: bash opencood/tools/run_noise_sweep.sh <model_dir>"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/.local/micromamba/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
ENV_NAME="${ENV_NAME:-heal}"

FUSION_METHOD="${FUSION_METHOD:-intermediate}"
SWEEP_MODE="${SWEEP_MODE:-paired}"
POS_STD_LIST="${POS_STD_LIST:-0,1,2,3,4}"
ROT_STD_LIST="${ROT_STD_LIST:-0,1,2,3,4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"

GPU_ALL="${GPU_ALL:-0}"
GPU_NONEGO="${GPU_NONEGO:-1}"
NOTE_PREFIX="${NOTE_PREFIX:-_final}"

MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"

COMMON_ARGS=(
  opencood/tools/inference_w_noise.py
  --model_dir "${MODEL_DIR}"
  --fusion_method "${FUSION_METHOD}"
  --pos-std-list "${POS_STD_LIST}"
  --rot-std-list "${ROT_STD_LIST}"
  --sweep-mode "${SWEEP_MODE}"
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
)

if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
fi

LOG_ROOT="${MODEL_DIR}/noise_sweep_logs"
mkdir -p "${LOG_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ALL="${LOG_ROOT}/noise_sweep_all_${STAMP}.log"
LOG_NONEGO="${LOG_ROOT}/noise_sweep_nonego_${STAMP}.log"

echo "[run_noise_sweep] MODEL_DIR=${MODEL_DIR}"
echo "[run_noise_sweep] POS_STD_LIST=${POS_STD_LIST} ROT_STD_LIST=${ROT_STD_LIST} SWEEP_MODE=${SWEEP_MODE}"
echo "[run_noise_sweep] GPU_ALL=${GPU_ALL} GPU_NONEGO=${GPU_NONEGO}"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ALL}" MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
  "${MICROMAMBA_BIN}" run -n "${ENV_NAME}" \
  python "${COMMON_ARGS[@]}" --noise-target all --note "${NOTE_PREFIX}_all" \
  2>&1 | tee "${LOG_ALL}" &
PID_ALL=$!

CUDA_VISIBLE_DEVICES="${GPU_NONEGO}" MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
  "${MICROMAMBA_BIN}" run -n "${ENV_NAME}" \
  python "${COMMON_ARGS[@]}" --noise-target non-ego --note "${NOTE_PREFIX}_nonego" \
  2>&1 | tee "${LOG_NONEGO}" &
PID_NONEGO=$!

wait "${PID_ALL}"
RC_ALL=$?
wait "${PID_NONEGO}"
RC_NONEGO=$?

set -e
if [[ "${RC_ALL}" -ne 0 || "${RC_NONEGO}" -ne 0 ]]; then
  echo "[run_noise_sweep] ERROR: one or more sweep jobs failed (all=${RC_ALL}, nonego=${RC_NONEGO})"
  exit 1
fi

echo "[run_noise_sweep] Done."
echo "[run_noise_sweep] Logs:"
echo "  ${LOG_ALL}"
echo "  ${LOG_NONEGO}"
echo "[run_noise_sweep] Result YAMLs will be named like:"
echo "  ${MODEL_DIR}/AP030507_none${NOTE_PREFIX}_all.yaml"
echo "  ${MODEL_DIR}/AP030507_none${NOTE_PREFIX}_nonego.yaml"

