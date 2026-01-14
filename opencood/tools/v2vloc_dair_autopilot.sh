#!/usr/bin/env bash
set -euo pipefail

# Simple DAIR autopilot:
#  1) Wait for a given CLEAN run to finish.
#  2) Ensure a full noise sweep exists for the CLEAN run.
#  3) Launch a NOISE fine-tune run (non-ego noise) initialized from CLEAN.
#  4) Auto-run noise sweep on NOISE run when training finishes (via watchdog).
#
# Usage:
#   bash opencood/tools/v2vloc_dair_autopilot.sh <clean_model_dir> [noise_yaml] [noise_run_dir]

CLEAN_MODEL_DIR="${1:-}"
NOISE_YAML="${2:-opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_pastat_noise1_nonego.yaml}"
NOISE_MODEL_DIR="${3:-}"

if [[ -z "${CLEAN_MODEL_DIR}" ]]; then
  echo "Usage: bash opencood/tools/v2vloc_dair_autopilot.sh <clean_model_dir> [noise_yaml] [noise_run_dir]"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SLEEP_SEC="${SLEEP_SEC:-60}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7,8,9}"
NPROC="${NPROC:-6}"
NUM_WORKERS="${NUM_WORKERS:-2}"

GPU_SWEEP_ALL="${GPU_SWEEP_ALL:-0}"
GPU_SWEEP_NONEGO="${GPU_SWEEP_NONEGO:-1}"

NOTE_PREFIX_CLEAN="${NOTE_PREFIX_CLEAN:-_final_clean}"
NOTE_PREFIX_NOISE="${NOTE_PREFIX_NOISE:-_final_noiseft}"

if [[ -z "${NOISE_MODEL_DIR}" ]]; then
  TS="$(date +%Y_%m_%d_%H_%M_%S)"
  NOISE_MODEL_DIR="opencood/logs/HeterBaseline_DAIR_lidar_pastat_noise1_nonego_ft_${TS}"
fi

echo "[autopilot] ROOT_DIR=${ROOT_DIR}"
echo "[autopilot] CLEAN_MODEL_DIR=${CLEAN_MODEL_DIR}"
echo "[autopilot] NOISE_YAML=${NOISE_YAML}"
echo "[autopilot] NOISE_MODEL_DIR=${NOISE_MODEL_DIR}"
echo "[autopilot] TRAIN_GPUS=${TRAIN_GPUS} NPROC=${NPROC} NUM_WORKERS=${NUM_WORKERS}"

TRAIN_LOG="${CLEAN_MODEL_DIR}/train_stdout.log"

if command -v rg >/dev/null 2>&1; then
  GREP_CMD=(rg -q)
else
  GREP_CMD=(grep -Eq)
fi

echo "[autopilot] waiting for CLEAN training to finish..."
while true; do
  finished=0
  if [[ -f "${TRAIN_LOG}" ]]; then
    if "${GREP_CMD[@]}" "Training Finished" "${TRAIN_LOG}"; then
      finished=1
    fi
  fi
  # also require the train_ddp process to be gone, to avoid racing with file writes.
  RAW_PATTERN="opencood/tools/train_ddp\\.py.*--model_dir ${CLEAN_MODEL_DIR//\//\\/}"
  if [[ "${finished}" -eq 1 ]] && ! ps -eo cmd | "${GREP_CMD[@]}" "${RAW_PATTERN}"; then
    echo "[autopilot] CLEAN finished."
    break
  fi
  sleep "${SLEEP_SEC}"
done

echo "[autopilot] ensuring CLEAN noise sweep exists..."
if [[ ! -f "${CLEAN_MODEL_DIR}/AP030507_none${NOTE_PREFIX_CLEAN}_all.yaml" || ! -f "${CLEAN_MODEL_DIR}/AP030507_none${NOTE_PREFIX_CLEAN}_nonego.yaml" ]]; then
  echo "[autopilot] running CLEAN noise sweep..."
  GPU_ALL="${GPU_SWEEP_ALL}" GPU_NONEGO="${GPU_SWEEP_NONEGO}" NOTE_PREFIX="${NOTE_PREFIX_CLEAN}" \
    bash opencood/tools/run_noise_sweep.sh "${CLEAN_MODEL_DIR}"
else
  echo "[autopilot] CLEAN sweep already present."
fi

echo "[autopilot] launching NOISE fine-tune run (init from CLEAN)..."
mkdir -p "${NOISE_MODEL_DIR}"

# Background watchdog (train + auto sweep on finish).
nohup python3 -u opencood/tools/train_watchdog.py \
  --yaml "${NOISE_YAML}" \
  --model_dir "${NOISE_MODEL_DIR}" \
  --init_model_dir "${CLEAN_MODEL_DIR}" \
  --fusion_method intermediate \
  --cuda_visible_devices "${TRAIN_GPUS}" \
  --nproc "${NPROC}" \
  --num_workers "${NUM_WORKERS}" \
  --half \
  --no_test \
  --run_sweep_on_finish \
  > "${NOISE_MODEL_DIR}/watchdog.log" 2>&1 &
echo $! > "${NOISE_MODEL_DIR}/watchdog.pid"

# Background status logger.
nohup python3 -u opencood/tools/live_train_status.py \
  --model_dir "${NOISE_MODEL_DIR}" \
  --interval 30 \
  --include_gpu \
  --out "${NOISE_MODEL_DIR}/live_status.jsonl" \
  > "${NOISE_MODEL_DIR}/live_status_runner.log" 2>&1 &
echo $! > "${NOISE_MODEL_DIR}/live_status_runner.pid"

echo "[autopilot] NOISE run started."
echo "[autopilot] tail -f ${NOISE_MODEL_DIR}/live_status.jsonl"
echo "[autopilot] tail -f ${NOISE_MODEL_DIR}/watchdog.log"

