#!/bin/bash
#SBATCH --job-name=final_pipeline
#SBATCH --output=logs/final_pipeline_%j.out
#SBATCH --error=logs/final_pipeline_%j.err
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

OUT_ROOT="${PROJECT_ROOT}/outputs/final"
NPY_BASE_ROOT="${PROJECT_ROOT}/output_npy/final"

PREPROCESS="${PREPROCESS:-median_filter_rms}"
TARGET_RMS="${TARGET_RMS:-0.15}"
RMS_SCOPE="${RMS_SCOPE:-site}"
TARGET_FS="${TARGET_FS:-1000}"
FILTER_ORDER_LOW="${FILTER_ORDER_LOW:-3}"
FILTER_DECAY_LOW="${FILTER_DECAY_LOW:-3}"
FILTER_ORDER_HIGH="${FILTER_ORDER_HIGH:-1}"
FILTER_DECAY_HIGH="${FILTER_DECAY_HIGH:-1}"
FILTER_SPEC_POHANG="${FILTER_SPEC_POHANG:-none:50}"
FILTER_SPEC_UTAH2019="${FILTER_SPEC_UTAH2019:-none:200}"
FILTER_SPEC_UTAH2023="${FILTER_SPEC_UTAH2023:-none:500}"
LOG_BASE="${LOG_BASE:-1.0}"
LOG_SMOOTH_SIGMA_CHANNEL="${LOG_SMOOTH_SIGMA_CHANNEL:-1.0}"
LOG_SMOOTH_SIGMA_TIME="${LOG_SMOOTH_SIGMA_TIME:-0.5}"
LOG_EPS="${LOG_EPS:-1e-8}"

format_token_number() {
  local value="$1"
  value="${value//- /m}"
  value="${value//-/m}"
  value="${value//./p}"
  echo "${value}"
}

filter_token() {
  local spec="$1"
  local low="${spec%%:*}"
  local high="${spec##*:}"
  if [[ "${low}" == "none" && "${high}" == "none" ]]; then
    echo "nofilter"
  elif [[ "${low}" == "none" ]]; then
    echo "lp$(format_token_number "${high}")hz"
  elif [[ "${high}" == "none" ]]; then
    echo "hp$(format_token_number "${low}")hz"
  else
    echo "bp$(format_token_number "${low}")hz-$(format_token_number "${high}")hz"
  fi
}

if [[ -n "${PREPROCESS_RUN_NAME:-}" ]]; then
  RUN_NAME="${PREPROCESS_RUN_NAME}"
else
  RUN_NAME="${PREPROCESS}__fs$(format_token_number "${TARGET_FS}")__rms$(format_token_number "${TARGET_RMS}")__scope-${RMS_SCOPE}__ordlo$(format_token_number "${FILTER_ORDER_LOW}")__ordhi$(format_token_number "${FILTER_ORDER_HIGH}")__pohang_$(filter_token "${FILTER_SPEC_POHANG}")__ut2019_$(filter_token "${FILTER_SPEC_UTAH2019}")__ut2023_$(filter_token "${FILTER_SPEC_UTAH2023}")"
fi

NPY_ROOT="${NPY_BASE_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_ROOT}" "${NPY_ROOT}"

INPUT_ROOT="${PROJECT_ROOT}/outputs"
MERGED_INV="${OUT_ROOT}/inventory_all.csv"
INV_REPORT="${OUT_ROOT}/inventory_merge_report.json"
PLAN_REPORT="${OUT_ROOT}/segment_plan_merge_report.json"
VALIDATION_REPORT="${OUT_ROOT}/validation_report.json"
VALIDATION_ERRORS="${OUT_ROOT}/validation_errors.csv"
ALL_NPY_METADATA="${NPY_ROOT}/metadata/all_samples.csv"
EXPERIMENT_OUT_DIR="${OUT_ROOT}/experiments/${RUN_NAME}"

echo "[CONFIG] PREPROCESS=${PREPROCESS}"
echo "[CONFIG] TARGET_RMS=${TARGET_RMS}"
echo "[CONFIG] RMS_SCOPE=${RMS_SCOPE}"
echo "[CONFIG] TARGET_FS=${TARGET_FS}"
echo "[CONFIG] FILTER_ORDER_LOW=${FILTER_ORDER_LOW}"
echo "[CONFIG] FILTER_DECAY_LOW=${FILTER_DECAY_LOW}"
echo "[CONFIG] FILTER_ORDER_HIGH=${FILTER_ORDER_HIGH}"
echo "[CONFIG] FILTER_DECAY_HIGH=${FILTER_DECAY_HIGH}"
echo "[CONFIG] FILTER_SPEC_POHANG=${FILTER_SPEC_POHANG}"
echo "[CONFIG] FILTER_SPEC_UTAH2019=${FILTER_SPEC_UTAH2019}"
echo "[CONFIG] FILTER_SPEC_UTAH2023=${FILTER_SPEC_UTAH2023}"
echo "[CONFIG] LOG_BASE=${LOG_BASE}"
echo "[CONFIG] LOG_SMOOTH_SIGMA_CHANNEL=${LOG_SMOOTH_SIGMA_CHANNEL}"
echo "[CONFIG] LOG_SMOOTH_SIGMA_TIME=${LOG_SMOOTH_SIGMA_TIME}"
echo "[CONFIG] LOG_EPS=${LOG_EPS}"
echo "[CONFIG] RUN_NAME=${RUN_NAME}"
echo "[CONFIG] NPY_ROOT=${NPY_ROOT}"

python -m datamaker.final.merge_inventories \
  --input_root "${INPUT_ROOT}" \
  --out_csv "${MERGED_INV}" \
  --out_report_json "${INV_REPORT}"

python -m datamaker.final.merge_segment_plans \
  --input_root "${INPUT_ROOT}" \
  --out_dir "${OUT_ROOT}" \
  --out_report_json "${PLAN_REPORT}"

python -m datamaker.final.validate_merged_plans \
  --plan_dir "${OUT_ROOT}" \
  --out_report_json "${VALIDATION_REPORT}" \
  --out_error_csv "${VALIDATION_ERRORS}"

python -m datamaker.final.build_npy_dataset \
  --csv "${OUT_ROOT}/segment_plan_event.csv" \
  --csv "${OUT_ROOT}/segment_plan_noise.csv" \
  --csv "${OUT_ROOT}/segment_plan_unlabel.csv" \
  --out_root "${NPY_BASE_ROOT}" \
  --dtype float32 \
  --target_fs "${TARGET_FS}" \
  --site_col site \
  --preprocess "${PREPROCESS}" \
  --target_rms "${TARGET_RMS}" \
  --rms_scope "${RMS_SCOPE}" \
  --filter_order_low "${FILTER_ORDER_LOW}" \
  --filter_decay_low "${FILTER_DECAY_LOW}" \
  --filter_order_high "${FILTER_ORDER_HIGH}" \
  --filter_decay_high "${FILTER_DECAY_HIGH}" \
  --filter_spec "pohang:${FILTER_SPEC_POHANG}" \
  --filter_spec "utah_2019:${FILTER_SPEC_UTAH2019}" \
  --filter_spec "utah_2023:${FILTER_SPEC_UTAH2023}" \
  --log_base "${LOG_BASE}" \
  --log_smooth_sigma_channel "${LOG_SMOOTH_SIGMA_CHANNEL}" \
  --log_smooth_sigma_time "${LOG_SMOOTH_SIGMA_TIME}" \
  --log_eps "${LOG_EPS}" \
  --append_preprocess_run_name \
  --preprocess_run_name "${RUN_NAME}"

python -m datamaker.final.make_experiment_splits_3site \
  --all_csv "${ALL_NPY_METADATA}" \
  --out_dir "${EXPERIMENT_OUT_DIR}" \
  --sites pohang utah_2019 utah_2023
