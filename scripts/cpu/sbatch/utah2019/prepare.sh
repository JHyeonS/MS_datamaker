#!/bin/bash
#SBATCH --job-name=ut19_prepare
#SBATCH --output=logs/ut19_prepare_%j.out
#SBATCH --error=logs/ut19_prepare_%j.err
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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT_ROOT="${PROJECT_ROOT}/outputs/utah2019"
mkdir -p "${OUT_ROOT}"

# TODO: set these to your real paths
EVENT_DIR="${PROJECT_ROOT}/data/utah2019/event"
NOISE_DIR="${PROJECT_ROOT}/data/utah2019/noise"
UNLABEL_DIR="${PROJECT_ROOT}/data/utah2019/unlabel"
LOWER_LABEL_CSV="${PROJECT_ROOT}/data/utah2019/utah_lower_labels_2s.csv"
UPPER_LABEL_CSV="${PROJECT_ROOT}/data/utah2019/utah_upper_labels_2s.csv"

INVENTORY_CSV="${OUT_ROOT}/inventory.csv"
EVENT_PLAN_CSV="${OUT_ROOT}/segment_plan_event.csv"
NOISE_PLAN_CSV="${OUT_ROOT}/segment_plan_noise.csv"
UNLABEL_PLAN_CSV="${OUT_ROOT}/segment_plan_unlabel.csv"

python -m src.datamaker.utah2019.make_inventory_utah2019   --event_dir "${EVENT_DIR}"   --noise_dir "${NOISE_DIR}"   --unlabel_dir "${UNLABEL_DIR}"   --out_csv "${INVENTORY_CSV}"

python -m src.datamaker.utah2019.make_segment_plan_event_utah2019   --inventory_csv "${INVENTORY_CSV}"   --lower_label_csv "${LOWER_LABEL_CSV}"   --upper_label_csv "${UPPER_LABEL_CSV}"   --out_csv "${EVENT_PLAN_CSV}"

python -m src.datamaker.utah2019.make_segment_plan_noise_unlabel_utah2019   --inventory_csv "${INVENTORY_CSV}"   --out_noise_csv "${NOISE_PLAN_CSV}"   --out_unlabel_csv "${UNLABEL_PLAN_CSV}"
