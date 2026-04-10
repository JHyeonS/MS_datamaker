#!/bin/bash
set -euo pipefail

ROOT=/home/ted1204/MS_Detection_Datamaker/MS_datamaker_FINAL
INPUT_ROOT=${ROOT}/input
OUTPUT_ROOT=${ROOT}/output
SCRIPT_ROOT=${ROOT}/scripts

mkdir -p ${OUTPUT_ROOT}
mkdir -p ${OUTPUT_ROOT}/reports

echo "========== STEP 1: merge inventories =========="
python ${SCRIPT_ROOT}/merge_inventories.py \
  --input_root ${INPUT_ROOT} \
  --out_csv ${OUTPUT_ROOT}/merged_inventory.csv \
  --out_report_json ${OUTPUT_ROOT}/reports/merge_inventory_report.json

echo "========== STEP 2: merge segment plans =========="
python ${SCRIPT_ROOT}/merge_segment_plans.py \
  --input_root ${INPUT_ROOT} \
  --out_dir ${OUTPUT_ROOT} \
  --out_report_json ${OUTPUT_ROOT}/reports/merge_segment_plan_report.json

echo "========== STEP 3: validate merged plans =========="
python ${SCRIPT_ROOT}/validate_merged_plans.py \
  --plan_dir ${OUTPUT_ROOT} \
  --out_report_json ${OUTPUT_ROOT}/reports/validation_report.json \
  --out_error_csv ${OUTPUT_ROOT}/reports/validation_errors.csv

echo "========== DONE =========="