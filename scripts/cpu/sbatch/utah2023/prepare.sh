#!/bin/bash
#SBATCH --job-name=ut23_prepare
#SBATCH --output=logs/ut23_prepare_%j.out
#SBATCH --error=logs/ut23_prepare_%j.err
#SBATCH --partition=v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate MS_datamaker

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Utah 2023 script is dataset_root-driven
RAW_DIR="${UTAH2023_RAW_DIR:-${PROJECT_ROOT}/MS_datamaker_UTAH_2023/data/event_2417}"
LABEL_CSV="${UTAH2023_LABEL_CSV:-${PROJECT_ROOT}/MS_datamaker_UTAH_2023/csv/utah_2023_labels_2s.csv}"
OUT_ROOT="${PROJECT_ROOT}/outputs/utah2023"
mkdir -p "${OUT_ROOT}"

python -m datamaker.utah2023.make_inventory_utah2023 \
  --raw_dir "${RAW_DIR}" \
  --out_csv "${OUT_ROOT}/inventory.csv" \
  --site utah_2023 \
  --view utah_2023 \
  --original_fs 1000 \
  --ch_start 400 \
  --ch_end 805

python -m datamaker.utah2023.make_segment_plan_from_labels_utah2023 \
  --inventory_csv "${OUT_ROOT}/inventory.csv" \
  --label_csv "${LABEL_CSV}" \
  --out_event_csv "${OUT_ROOT}/segment_plan_event.csv" \
  --out_noise_csv "${OUT_ROOT}/segment_plan_noise.csv" \
  --out_unlabel_csv "${OUT_ROOT}/segment_plan_unlabel.csv" \
  --out_all_csv "${OUT_ROOT}/segment_plan_all.csv"
