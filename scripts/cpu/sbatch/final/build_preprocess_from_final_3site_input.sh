#!/bin/bash
#SBATCH --job-name=final_preprocess
#SBATCH --output=logs/final_preprocess_%j.out
#SBATCH --error=logs/final_preprocess_%j.err
#SBATCH --partition=v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate MS_datamaker

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PLAN_ROOT="${PLAN_ROOT:-${PROJECT_ROOT}/outputs/final_3site_input}"
NPY_BASE_ROOT="${PROJECT_ROOT}/output_npy/final"

PREPROCESS="${PREPROCESS:-filter_logenv_rms}"
TARGET_RMS="${TARGET_RMS:-0.15}"
RMS_SCOPE="${RMS_SCOPE:-site}"
TARGET_FS="${TARGET_FS:-1000}"
FILTER_ORDER_LOW="${FILTER_ORDER_LOW:-3}"
FILTER_DECAY_LOW="${FILTER_DECAY_LOW:-3}"
FILTER_ORDER_HIGH="${FILTER_ORDER_HIGH:-1}"
FILTER_DECAY_HIGH="${FILTER_DECAY_HIGH:-1}"
FILTER_SPEC_POHANG="${FILTER_SPEC_POHANG:-none:50}"
FILTER_SPEC_UTAH2019="${FILTER_SPEC_UTAH2019:-none:50}"
FILTER_SPEC_UTAH2023="${FILTER_SPEC_UTAH2023:-none:50}"
LOG_BASE="${LOG_BASE:-1.0}"
LOG_SMOOTH_SIGMA_CHANNEL="${LOG_SMOOTH_SIGMA_CHANNEL:-1.0}"
LOG_SMOOTH_SIGMA_TIME="${LOG_SMOOTH_SIGMA_TIME:-0.5}"
LOG_EPS="${LOG_EPS:-1e-8}"
OVERWRITE="${OVERWRITE:-0}"

if [[ -z "${PREPROCESS_RUN_NAME:-}" ]]; then
  echo "[ERROR] PREPROCESS_RUN_NAME is required" >&2
  exit 2
fi

NPY_ROOT="${NPY_BASE_ROOT}/${PREPROCESS_RUN_NAME}"
ALL_NPY_METADATA="${NPY_ROOT}/metadata/all_samples.csv"
EXPERIMENT_OUT_DIR="${NPY_ROOT}/metadata/experiments"

BUILD_ARGS=(
  --csv "${PLAN_ROOT}/segment_plan_event.csv"
  --csv "${PLAN_ROOT}/segment_plan_noise.csv"
  --csv "${PLAN_ROOT}/segment_plan_unlabel.csv"
  --out_root "${NPY_BASE_ROOT}"
  --dtype float32
  --target_fs "${TARGET_FS}"
  --site_col site
  --preprocess "${PREPROCESS}"
  --target_rms "${TARGET_RMS}"
  --rms_scope "${RMS_SCOPE}"
  --filter_order_low "${FILTER_ORDER_LOW}"
  --filter_decay_low "${FILTER_DECAY_LOW}"
  --filter_order_high "${FILTER_ORDER_HIGH}"
  --filter_decay_high "${FILTER_DECAY_HIGH}"
  --filter_spec "pohang:${FILTER_SPEC_POHANG}"
  --filter_spec "utah_2019:${FILTER_SPEC_UTAH2019}"
  --filter_spec "utah_2023:${FILTER_SPEC_UTAH2023}"
  --log_base "${LOG_BASE}"
  --log_smooth_sigma_channel "${LOG_SMOOTH_SIGMA_CHANNEL}"
  --log_smooth_sigma_time "${LOG_SMOOTH_SIGMA_TIME}"
  --log_eps "${LOG_EPS}"
  --append_preprocess_run_name
  --preprocess_run_name "${PREPROCESS_RUN_NAME}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
fi

echo "[CONFIG] PLAN_ROOT=${PLAN_ROOT}"
echo "[CONFIG] PREPROCESS_RUN_NAME=${PREPROCESS_RUN_NAME}"
echo "[CONFIG] PREPROCESS=${PREPROCESS}"
echo "[CONFIG] TARGET_FS=${TARGET_FS}"
echo "[CONFIG] TARGET_RMS=${TARGET_RMS}"
echo "[CONFIG] RMS_SCOPE=${RMS_SCOPE}"
echo "[CONFIG] FILTER_SPEC_POHANG=${FILTER_SPEC_POHANG}"
echo "[CONFIG] FILTER_SPEC_UTAH2019=${FILTER_SPEC_UTAH2019}"
echo "[CONFIG] FILTER_SPEC_UTAH2023=${FILTER_SPEC_UTAH2023}"
echo "[CONFIG] LOG_BASE=${LOG_BASE}"
echo "[CONFIG] LOG_SMOOTH_SIGMA_CHANNEL=${LOG_SMOOTH_SIGMA_CHANNEL}"
echo "[CONFIG] LOG_SMOOTH_SIGMA_TIME=${LOG_SMOOTH_SIGMA_TIME}"
echo "[CONFIG] LOG_EPS=${LOG_EPS}"
echo "[CONFIG] NPY_ROOT=${NPY_ROOT}"

python -m datamaker.final.validate_merged_plans \
  --plan_dir "${PLAN_ROOT}" \
  --out_report_json "${NPY_ROOT}/metadata/input_validation_report.json" \
  --out_error_csv "${NPY_ROOT}/metadata/input_validation_errors.csv"

python -m datamaker.final.build_npy_dataset "${BUILD_ARGS[@]}"

python -m datamaker.final.make_experiment_splits_3site \
  --all_csv "${ALL_NPY_METADATA}" \
  --out_dir "${EXPERIMENT_OUT_DIR}" \
  --sites pohang utah_2019 utah_2023
