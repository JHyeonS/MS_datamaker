#!/bin/bash
#SBATCH --job-name=ph_prepare
#SBATCH --output=logs/ph_prepare_%j.out
#SBATCH --error=logs/ph_prepare_%j.err
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

OUT_ROOT="${PROJECT_ROOT}/outputs/pohang"
mkdir -p "${OUT_ROOT}"

# TODO: set these to your real paths
EVENT_DIR="${PROJECT_ROOT}/data/pohang/event"
NOISE_DIR="${PROJECT_ROOT}/data/pohang/noise"
UNLABEL_DIR="${PROJECT_ROOT}/data/pohang/unlabel"
LABEL_CSV="${PROJECT_ROOT}/data/pohang/pohang_labels_2s.csv"

INVENTORY_CSV="${OUT_ROOT}/inventory.csv"
EVENT_PLAN_CSV="${OUT_ROOT}/segment_plan_event.csv"
NOISE_PLAN_CSV="${OUT_ROOT}/segment_plan_noise.csv"
UNLABEL_PLAN_CSV="${OUT_ROOT}/segment_plan_unlabel.csv"

python -m datamaker.pohang.make_file_inventory   --event_dir "${EVENT_DIR}"   --noise_dir "${NOISE_DIR}"   --unlabel_dir "${UNLABEL_DIR}"   --out_csv "${INVENTORY_CSV}"   --site pohang

python -m datamaker.pohang.make_segment_plan_event_pohang   --inventory_csv "${INVENTORY_CSV}"   --label_csv "${LABEL_CSV}"   --out_csv "${EVENT_PLAN_CSV}"

python -m datamaker.pohang.make_segment_plan_noise_unlabel_pohang   --inventory_csv "${INVENTORY_CSV}"   --out_noise_csv "${NOISE_PLAN_CSV}"   --out_unlabel_csv "${UNLABEL_PLAN_CSV}"
