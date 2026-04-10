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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT_ROOT="${PROJECT_ROOT}/outputs/final"
NPY_ROOT="${PROJECT_ROOT}/output_npy/final"
mkdir -p "${OUT_ROOT}" "${NPY_ROOT}"

PH_INV="${PROJECT_ROOT}/outputs/pohang/inventory.csv"
UT19_INV="${PROJECT_ROOT}/outputs/utah2019/inventory.csv"
UT23_INV="${PROJECT_ROOT}/outputs/utah2023/inventory.csv"

PH_EVENT="${PROJECT_ROOT}/outputs/pohang/segment_plan_event.csv"
PH_NOISE="${PROJECT_ROOT}/outputs/pohang/segment_plan_noise.csv"
PH_UNLABEL="${PROJECT_ROOT}/outputs/pohang/segment_plan_unlabel.csv"

UT19_EVENT="${PROJECT_ROOT}/outputs/utah2019/segment_plan_event.csv"
UT19_NOISE="${PROJECT_ROOT}/outputs/utah2019/segment_plan_noise.csv"
UT19_UNLABEL="${PROJECT_ROOT}/outputs/utah2019/segment_plan_unlabel.csv"

UT23_PLAN="${PROJECT_ROOT}/outputs/utah2023/segment_plan.csv"

MERGED_INV="${OUT_ROOT}/inventory_all.csv"
MERGED_PLAN="${OUT_ROOT}/all_samples.csv"

python -m src.datamaker.final.merge_inventories   --inputs "${PH_INV}" "${UT19_INV}" "${UT23_INV}"   --output "${MERGED_INV}"

python -m src.datamaker.final.merge_segment_plans   --inputs     "${PH_EVENT}" "${PH_NOISE}" "${PH_UNLABEL}"     "${UT19_EVENT}" "${UT19_NOISE}" "${UT19_UNLABEL}"     "${UT23_PLAN}"   --output "${MERGED_PLAN}"

python -m src.datamaker.final.validate_merged_plans   --inventory_csv "${MERGED_INV}"   --samples_csv "${MERGED_PLAN}"

python -m src.datamaker.final.build_npy_dataset   --csv "${PH_EVENT}"   --csv "${PH_NOISE}"   --csv "${PH_UNLABEL}"   --csv "${UT19_EVENT}"   --csv "${UT19_NOISE}"   --csv "${UT19_UNLABEL}"   --csv "${UT23_PLAN}"   --out_root "${NPY_ROOT}"   --dtype float32   --target_fs 1000   --site_col site

python -m src.datamaker.final.make_experiment_splits_3site   --all_csv "${MERGED_PLAN}"   --out_dir "${OUT_ROOT}/experiments"   --sites pohang utah2019 utah2023
