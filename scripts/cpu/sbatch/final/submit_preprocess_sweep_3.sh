#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"

PIPELINE="scripts/cpu/sbatch/final/full_pipeline.sh"

submit_one() {
  local name="$1"
  local target_fs="$2"
  local target_rms="$3"
  local pohang_filter="$4"
  local ut2019_filter="$5"
  local ut2023_filter="$6"
  local order_low="$7"
  local order_high="$8"

  echo "[SUBMIT] ${name}"
  echo "         TARGET_FS=${target_fs} TARGET_RMS=${target_rms}"
  echo "         FILTER_SPEC_POHANG=${pohang_filter}"
  echo "         FILTER_SPEC_UTAH2019=${ut2019_filter}"
  echo "         FILTER_SPEC_UTAH2023=${ut2023_filter}"
  echo "         FILTER_ORDER_LOW=${order_low} FILTER_ORDER_HIGH=${order_high}"

  local export_vars
  export_vars="ALL"
  export_vars+=",PREPROCESS_RUN_NAME=${name}"
  export_vars+=",PREPROCESS=median_filter_rms"
  export_vars+=",TARGET_FS=${target_fs}"
  export_vars+=",TARGET_RMS=${target_rms}"
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

# Experiment 1: current fixed-preprocess idea.
# AGC off, site-specific low-pass filters, site RMS normalization.
submit_one \
  "sweep01_current_fs1000_rms0p15_phlp50_ut19lp200_ut23lp500" \
  "1000" \
  "0.15" \
  "none:50" \
  "none:200" \
  "none:500" \
  "3" \
  "1"

# Experiment 2: linear midpoint between current and old model-front settings.
# Treat old bandpass_low=3 as low cutoff target and old bandpass_high=50 as high cutoff target.
# Midpoint uses low=1.5 Hz, site high cutoffs halfway toward 50 Hz, fs halfway toward 2000 Hz,
# and filter high-order halfway from 1 to 4.
submit_one \
  "sweep02_mid_fs1500_rms0p15_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275" \
  "1500" \
  "0.15" \
  "1.5:50" \
  "1.5:125" \
  "1.5:275" \
  "3.5" \
  "2.5"

# Experiment 3: old preprocessing frequency/sampling target translated into dataset-generation preprocessing.
# AGC remains off. normalize=robust is intentionally not used; RMS normalization is kept for this workflow.
submit_one \
  "sweep03_oldfreq_fs2000_rms0p15_bp3-50" \
  "2000" \
  "0.15" \
  "3:50" \
  "3:50" \
  "3:50" \
  "4" \
  "4"
