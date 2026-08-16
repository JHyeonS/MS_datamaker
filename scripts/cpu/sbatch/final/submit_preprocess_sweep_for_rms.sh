#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <target_rms>" >&2
  echo "Example: $0 1.0" >&2
  exit 2
fi

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"

TARGET_RMS_VALUE="$1"
RMS_TOKEN="${TARGET_RMS_VALUE//./p}"
RMS_TOKEN="${RMS_TOKEN//-/m}"
PIPELINE="scripts/cpu/sbatch/final/full_pipeline.sh"

submit_one() {
  local name="$1"
  local target_fs="$2"
  local pohang_filter="$3"
  local ut2019_filter="$4"
  local ut2023_filter="$5"
  local order_low="$6"
  local order_high="$7"

  echo "[SUBMIT] ${name}"
  echo "         TARGET_FS=${target_fs} TARGET_RMS=${TARGET_RMS_VALUE}"
  echo "         FILTER_SPEC_POHANG=${pohang_filter}"
  echo "         FILTER_SPEC_UTAH2019=${ut2019_filter}"
  echo "         FILTER_SPEC_UTAH2023=${ut2023_filter}"
  echo "         FILTER_ORDER_LOW=${order_low} FILTER_ORDER_HIGH=${order_high}"

  local export_vars
  export_vars="ALL"
  export_vars+=",PREPROCESS_RUN_NAME=${name}"
  export_vars+=",PREPROCESS=median_filter_rms"
  export_vars+=",TARGET_FS=${target_fs}"
  export_vars+=",TARGET_RMS=${TARGET_RMS_VALUE}"
  export_vars+=",RMS_SCOPE=site"
  export_vars+=",FILTER_SPEC_POHANG=${pohang_filter}"
  export_vars+=",FILTER_SPEC_UTAH2019=${ut2019_filter}"
  export_vars+=",FILTER_SPEC_UTAH2023=${ut2023_filter}"
  export_vars+=",FILTER_ORDER_LOW=${order_low}"
  export_vars+=",FILTER_DECAY_LOW=3"
  export_vars+=",FILTER_ORDER_HIGH=${order_high}"
  export_vars+=",FILTER_DECAY_HIGH=1"

  sbatch \
    --job-name="${name}" \
    --export="${export_vars}" \
    "${PIPELINE}"
}

submit_one \
  "sweep01_current_fs1000_rms${RMS_TOKEN}_phlp50_ut19lp200_ut23lp500" \
  "1000" \
  "none:50" \
  "none:200" \
  "none:500" \
  "3" \
  "1"

submit_one \
  "sweep02_mid_fs1500_rms${RMS_TOKEN}_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275" \
  "1500" \
  "1.5:50" \
  "1.5:125" \
  "1.5:275" \
  "3.5" \
  "2.5"

submit_one \
  "sweep03_oldfreq_fs2000_rms${RMS_TOKEN}_bp3-50" \
  "2000" \
  "3:50" \
  "3:50" \
  "3:50" \
  "4" \
  "4"
