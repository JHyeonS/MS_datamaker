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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Utah 2023 script is dataset_root-driven
DATASET_ROOT="${PROJECT_ROOT}/data/utah2023"
OUT_ROOT="${PROJECT_ROOT}/outputs/utah2023"
mkdir -p "${OUT_ROOT}"

python -m src.datamaker.utah2023.merge_metadata_and_labels   --dataset_root "${DATASET_ROOT}"

python -m src.datamaker.utah2023.make_inventory_utah2023   --input_csv "${DATASET_ROOT}/metadata/all_samples.csv"   --out_csv "${OUT_ROOT}/inventory.csv"

python -m src.datamaker.utah2023.make_segment_plan_from_labels_utah2023   --inventory_csv "${OUT_ROOT}/inventory.csv"   --out_csv "${OUT_ROOT}/segment_plan.csv"

python -m src.datamaker.utah2023.sgy_to_npy_dataset   --segment_plan "${OUT_ROOT}/segment_plan.csv"   --output_dir "${PROJECT_ROOT}/output_npy/utah2023"
