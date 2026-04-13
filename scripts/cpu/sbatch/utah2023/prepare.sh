#!/usr/bin/env bash
set -euo pipefail

# Utah 2023 site-level prepare script for FINAL merge pipeline.
#
# What it does:
# 1) Build Utah 2023 inventory.csv from raw SGY/SEGY files
# 2) Build segment_plan_{event,noise,unlabel,all}.csv from label CSV
# 3) Copy outputs into MS_datamaker_FINAL/input/utah_2023/
#
# Usage (override paths as needed):
#   bash scripts/cpu/sbatch/utah2023/prepare.sh
#   RAW_DIR=/data/utah2023 LABEL_CSV=/data/utah_2023_labels_2s.csv \
#   bash scripts/cpu/sbatch/utah2023/prepare.sh

REPO_ROOT=${REPO_ROOT:-/home/ted1204/MS_datamaker}
SITE_ROOT=${SITE_ROOT:-${REPO_ROOT}/MS_datamaker_UTAH_2023}
FINAL_ROOT=${FINAL_ROOT:-${REPO_ROOT}/MS_datamaker_FINAL}

RAW_DIR=${RAW_DIR:-${SITE_ROOT}/data}
LABEL_CSV=${LABEL_CSV:-${SITE_ROOT}/csv/utah_2023_labels_2s.csv}

OUT_DIR=${OUT_DIR:-${SITE_ROOT}/outputs}
FINAL_INPUT_DIR=${FINAL_INPUT_DIR:-${FINAL_ROOT}/input/utah_2023}

SITE=${SITE:-utah_2023}
VIEW=${VIEW:-utah_2023_main}
ORIGINAL_FS=${ORIGINAL_FS:-1000}
CH_START=${CH_START:-400}
CH_END=${CH_END:-805}

INVENTORY_CSV=${INVENTORY_CSV:-${OUT_DIR}/inventory.csv}
SEG_EVENT_CSV=${SEG_EVENT_CSV:-${OUT_DIR}/segment_plan_event.csv}
SEG_NOISE_CSV=${SEG_NOISE_CSV:-${OUT_DIR}/segment_plan_noise.csv}
SEG_UNLABEL_CSV=${SEG_UNLABEL_CSV:-${OUT_DIR}/segment_plan_unlabel.csv}
SEG_ALL_CSV=${SEG_ALL_CSV:-${OUT_DIR}/segment_plan_all.csv}

PY_INVENTORY=${PY_INVENTORY:-${SITE_ROOT}/scripts/make_inventory_utah2023.py}
PY_SEGMENT=${PY_SEGMENT:-${SITE_ROOT}/scripts/make_segment_plan_from_labels_utah2023.py}

mkdir -p "${OUT_DIR}" "${FINAL_INPUT_DIR}"

if [[ ! -f "${PY_INVENTORY}" ]]; then
  echo "[ERROR] inventory builder not found: ${PY_INVENTORY}" >&2
  exit 1
fi

if [[ ! -f "${PY_SEGMENT}" ]]; then
  echo "[ERROR] segment-plan builder not found: ${PY_SEGMENT}" >&2
  exit 1
fi

if [[ ! -d "${RAW_DIR}" ]]; then
  echo "[ERROR] raw dir not found: ${RAW_DIR}" >&2
  exit 1
fi

if [[ ! -f "${LABEL_CSV}" ]]; then
  echo "[ERROR] label csv not found: ${LABEL_CSV}" >&2
  exit 1
fi

echo "========== UTAH 2023 | STEP 1: inventory =========="
python "${PY_INVENTORY}" \
  --raw_dir "${RAW_DIR}" \
  --out_csv "${INVENTORY_CSV}" \
  --site "${SITE}" \
  --view "${VIEW}" \
  --original_fs "${ORIGINAL_FS}" \
  --ch_start "${CH_START}" \
  --ch_end "${CH_END}"

echo "========== UTAH 2023 | STEP 2: segment plans =========="
python "${PY_SEGMENT}" \
  --inventory_csv "${INVENTORY_CSV}" \
  --label_csv "${LABEL_CSV}" \
  --out_event_csv "${SEG_EVENT_CSV}" \
  --out_noise_csv "${SEG_NOISE_CSV}" \
  --out_unlabel_csv "${SEG_UNLABEL_CSV}" \
  --out_all_csv "${SEG_ALL_CSV}"

echo "========== UTAH 2023 | STEP 3: export to FINAL input =========="
cp "${INVENTORY_CSV}" "${FINAL_INPUT_DIR}/inventory.csv"
cp "${SEG_EVENT_CSV}" "${FINAL_INPUT_DIR}/segment_plan_event.csv"
cp "${SEG_NOISE_CSV}" "${FINAL_INPUT_DIR}/segment_plan_noise.csv"
cp "${SEG_UNLABEL_CSV}" "${FINAL_INPUT_DIR}/segment_plan_unlabel.csv"
cp "${SEG_ALL_CSV}" "${FINAL_INPUT_DIR}/segment_plan_all.csv"

echo "[DONE] exported Utah 2023 artifacts to: ${FINAL_INPUT_DIR}"
