#!/usr/bin/env bash
set -euo pipefail

# Lightweight monitor for Slurm pose-sweep jobs + auto-plot when finished.
#
# Example:
#   bash HEAL/opencood/tools/watch_pose_sweep_jobs.sh \
#     --model-dir opencood/logs/<run_dir> \
#     --note _comm200_paper \
#     --jobs 143,144,145,146

MODEL_DIR=""
NOTE=""
JOBS=""
SLEEP_SEC="120"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2;;
    --note) NOTE="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --sleep) SLEEP_SEC="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "${MODEL_DIR}" || -z "${NOTE}" || -z "${JOBS}" ]]; then
  echo "Usage: $0 --model-dir <opencood/logs/...> --note <_tag> --jobs <id,id,...> [--sleep 120]" >&2
  exit 2
fi

REPO_ROOT="/home/qqxluca/v2xreg_private"
HEAL_DIR="${REPO_ROOT}/HEAL"
OUTDIR="${HEAL_DIR}/${MODEL_DIR}"

HEAL_PYTHON="${HEAL_PYTHON:-$HOME/.micromamba/envs/heal/bin/python}"
if [[ ! -x "${HEAL_PYTHON}" ]]; then
  echo "heal python not found/executable at: ${HEAL_PYTHON}" >&2
  exit 2
fi

Y_NONE="${OUTDIR}/AP030507_none${NOTE}.yaml"
Y_V2X="${OUTDIR}/AP030507_v2xregpp_initfree${NOTE}.yaml"
Y_FA_PAPER="${OUTDIR}/AP030507_freealign_paper${NOTE}.yaml"
Y_FA_REPO="${OUTDIR}/AP030507_freealign_repo${NOTE}.yaml"

echo "[watch_pose_sweep_jobs] model_dir=${MODEL_DIR}"
echo "[watch_pose_sweep_jobs] note=${NOTE}"
echo "[watch_pose_sweep_jobs] jobs=${JOBS}"
echo "[watch_pose_sweep_jobs] outdir=${OUTDIR}"

while true; do
  echo
  echo "==== $(date +'%F %T') ===="
  if command -v squeue >/dev/null 2>&1; then
    squeue -j "${JOBS}" -o "%.18i %.9P %.25j %.8u %.2t %.10M %.6D %R" || true
  fi

  echo "-- eval files (comm200_paper) --"
  (ls -1 "${OUTDIR}" | rg "^eval_.*${NOTE#_}" || true) | head -n 20

  echo "-- outputs --"
  for p in "${Y_NONE}" "${Y_V2X}" "${Y_FA_PAPER}" "${Y_FA_REPO}"; do
    if [[ -f "${p}" ]]; then
      echo "FOUND $(basename "${p}")"
    else
      echo "MISSING $(basename "${p}")"
    fi
  done

  if [[ -f "${Y_NONE}" && -f "${Y_V2X}" && -f "${Y_FA_PAPER}" && -f "${Y_FA_REPO}" ]]; then
    echo "[watch_pose_sweep_jobs] all YAMLs found; plotting..."
    OUT_PNG="${REPO_ROOT}/docs/operations/v2v4real_extrinsic_sweep${NOTE}_ap.png"
    "${HEAL_PYTHON}" "${HEAL_DIR}/opencood/tools/plot_pose_sweep.py" \
        --yamls "${Y_NONE}" "${Y_V2X}" "${Y_FA_PAPER}" "${Y_FA_REPO}" \
        --metric all --scale percent \
        --out "${OUT_PNG}" \
        --title "V2V4Real PASTAT: AP vs pose noise (${NOTE#_})"
    echo "[watch_pose_sweep_jobs] wrote: ${OUT_PNG}"
    exit 0
  fi

  sleep "${SLEEP_SEC}"
done
