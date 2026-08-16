#!/bin/bash
#SBATCH --job-name=raw_freq
#SBATCH --output=logs/raw_frequency_%j.out
#SBATCH --error=logs/raw_frequency_%j.err
#SBATCH --partition=v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=24G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate MS_datamaker

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PLAN_DIR="${PLAN_DIR:-outputs/final_3site_input}"
OUT_DIR="${OUT_DIR:-output_npy/final/raw_frequency_analysis}"
LOG_EVERY="${LOG_EVERY:-100}"

ARGS=(
  --plan_dir "${PLAN_DIR}"
  --out_dir "${OUT_DIR}"
  --log_every "${LOG_EVERY}"
)

if [[ -n "${SITE:-}" ]]; then
  ARGS+=(--site "${SITE}")
fi

if [[ -n "${LABEL:-}" ]]; then
  ARGS+=(--label "${LABEL}")
fi

if [[ -n "${MAX_ROWS:-}" ]]; then
  ARGS+=(--max_rows "${MAX_ROWS}")
fi

echo "[CONFIG] PLAN_DIR=${PLAN_DIR}"
echo "[CONFIG] OUT_DIR=${OUT_DIR}"
echo "[CONFIG] SITE=${SITE:-all}"
echo "[CONFIG] LABEL=${LABEL:-all}"
echo "[CONFIG] MAX_ROWS=${MAX_ROWS:-all}"

python -m datamaker.final.analyze_raw_frequency_energy "${ARGS[@]}"
