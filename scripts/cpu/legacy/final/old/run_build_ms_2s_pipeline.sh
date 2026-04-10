#!/bin/bash
#SBATCH --job-name=build_ms_2s
#SBATCH --output=/home/ted1204/MS_datamaker_merge/logs/build_ms_2s_%j.out
#SBATCH --error=/home/ted1204/MS_datamaker_merge/logs/build_ms_2s_%j.err
#SBATCH --partition=v3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate MS_datamaker

ROOT=/home/ted1204/MS_datamaker_merge
OUT=${ROOT}/outputs_2s
OUT_NPY=${ROOT}/outputs_npy_2s

mkdir -p ${ROOT}/logs
mkdir -p ${OUT}
mkdir -p ${OUT_NPY}

cd ${ROOT}

echo "========== STEP 1: 2s segment plans =========="
python src/make_segment_plan_event_2s.py
python src/make_segment_plan_noise_unlabel_2s.py

echo "========== STEP 2: build npy =========="
python src/build_npy_dataset.py   --csv ${OUT}/segment_plan_event.csv   --csv ${OUT}/segment_plan_noise.csv   --csv ${OUT}/segment_plan_unlabel.csv   --out_root ${OUT_NPY}   --dtype float32   --target_fs 1000   --site_fs pohang:1000   --site_fs utah_2019:1000   --site_col site

echo "========== DONE =========="
