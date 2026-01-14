#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "Usage: bash opencood/tools/watch_train_and_sweep.sh <model_dir>"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SLEEP_SEC="${SLEEP_SEC:-60}"

if [[ "${MODEL_DIR}" = /* ]]; then
  MODEL_DIR_ABS="${MODEL_DIR}"
else
  MODEL_DIR_ABS="${ROOT_DIR}/${MODEL_DIR}"
fi

TRAIN_LOG="${MODEL_DIR_ABS}/train_stdout.log"

if command -v rg >/dev/null 2>&1; then
  GREP_CMD=(rg -q)
else
  GREP_CMD=(grep -Eq)
fi

echo "[watch_train_and_sweep] watching: ${MODEL_DIR}"
echo "[watch_train_and_sweep] polling every ${SLEEP_SEC}s"

finished=0

while true; do
  if [[ -f "${TRAIN_LOG}" ]]; then
    if "${GREP_CMD[@]}" "Training Finished" "${TRAIN_LOG}"; then
      if [[ "${finished}" -eq 0 ]]; then
        echo "[watch_train_and_sweep] detected 'Training Finished' in ${TRAIN_LOG}"
        finished=1
      fi
    fi
  fi

  # Match both raw (relative) and absolute paths to be robust.
  RAW_PATTERN="opencood/tools/train_ddp\\.py.*--model_dir ${MODEL_DIR//\//\\/}"
  ABS_PATTERN="opencood/tools/train_ddp\\.py.*--model_dir ${MODEL_DIR_ABS//\//\\/}"
  if ! ps -eo cmd | "${GREP_CMD[@]}" "${RAW_PATTERN}|${ABS_PATTERN}"; then
    if [[ "${finished}" -eq 0 ]]; then
      echo "[watch_train_and_sweep] train_ddp process not found; proceeding to sweep"
    else
      echo "[watch_train_and_sweep] train_ddp process exited; starting sweep"
    fi
    break
  fi

  if [[ "${finished}" -eq 1 ]]; then
    echo "[watch_train_and_sweep] waiting for train_ddp to exit..."
  fi
  sleep "${SLEEP_SEC}"
done

echo "[watch_train_and_sweep] starting noise sweep"
bash opencood/tools/run_noise_sweep.sh "${MODEL_DIR}"
