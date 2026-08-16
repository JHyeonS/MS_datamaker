#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"

PIPELINE="scripts/cpu/sbatch/final/build_preprocess_from_final_3site_input.sh"
TARGET_RMS_VALUE="0.15"
TARGET_FS_VALUE="1000"
FILTER_SPEC="none:50"
LOG_BASE_VALUE="1.0"
LOG_SIGMA_CHANNEL="1.0"
LOG_SIGMA_TIME="0.5"
LOG_EPS_VALUE="1e-8"

submit_one() {
  local preprocess="$1"
  local name="$2"

  echo "[SUBMIT] ${name}"
  echo "         PREPROCESS=${preprocess}"
  echo "         TARGET_FS=${TARGET_FS_VALUE} TARGET_RMS=${TARGET_RMS_VALUE}"
  echo "         FILTER_SPEC_ALL=${FILTER_SPEC}"
  echo "         LOG_BASE=${LOG_BASE_VALUE} LOG_SIGMA=${LOG_SIGMA_CHANNEL},${LOG_SIGMA_TIME} LOG_EPS=${LOG_EPS_VALUE}"

  local export_vars
  export_vars="ALL"
  export_vars+=",PREPROCESS_RUN_NAME=${name}"
  export_vars+=",PREPROCESS=${preprocess}"
  export_vars+=",TARGET_FS=${TARGET_FS_VALUE}"
  export_vars+=",TARGET_RMS=${TARGET_RMS_VALUE}"
  export_vars+=",RMS_SCOPE=site"
  export_vars+=",FILTER_SPEC_POHANG=${FILTER_SPEC}"
  export_vars+=",FILTER_SPEC_UTAH2019=${FILTER_SPEC}"
  export_vars+=",FILTER_SPEC_UTAH2023=${FILTER_SPEC}"
  export_vars+=",FILTER_ORDER_LOW=3"
  export_vars+=",FILTER_DECAY_LOW=3"
  export_vars+=",FILTER_ORDER_HIGH=1"
  export_vars+=",FILTER_DECAY_HIGH=1"
  export_vars+=",LOG_BASE=${LOG_BASE_VALUE}"
  export_vars+=",LOG_SMOOTH_SIGMA_CHANNEL=${LOG_SIGMA_CHANNEL}"
  export_vars+=",LOG_SMOOTH_SIGMA_TIME=${LOG_SIGMA_TIME}"
  export_vars+=",LOG_EPS=${LOG_EPS_VALUE}"
  export_vars+=",PLAN_ROOT=${PROJECT_ROOT}/outputs/final_3site_input"
  export_vars+=",OVERWRITE=1"

  sbatch \
    --job-name="${name}" \
    --export="${export_vars}" \
    "${PIPELINE}"
}

submit_one \
  "filter_logenv_rms" \
  "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5"

submit_one \
  "filter_rms" \
  "visualbest_filter_rms_fs1000_rms0p15_lp50"
