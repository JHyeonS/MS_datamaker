#!/bin/bash
#SBATCH --job-name=build_npy
#SBATCH --output=/home/ted1204/MS_datamaker_merge/logs/build_npy_%j.out
#SBATCH --error=/home/ted1204/MS_datamaker_merge/logs/build_npy_%j.err
#SBATCH --partition=v3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate MS_datamaker

PROJECT_ROOT=/home/ted1204/MS_datamaker_merge
CSV_ROOT=${PROJECT_ROOT}/outputs
OUT_ROOT=${PROJECT_ROOT}/outputs_npy

mkdir -p ${PROJECT_ROOT}/logs
mkdir -p ${OUT_ROOT}

cd ${PROJECT_ROOT}

EVENT_CSV=${CSV_ROOT}/segment_plan_event.csv
NOISE_CSV=${CSV_ROOT}/segment_plan_noise.csv
UNLABEL_CSV=${CSV_ROOT}/segment_plan_unlabel.csv

echo "========== Dataset Build =========="
echo "HOST      : $(hostname)"
echo "PWD       : $(pwd)"
echo "PYTHON    : $(which python)"
echo "EVENT CSV : $EVENT_CSV"
echo "NOISE CSV : $NOISE_CSV"
echo "UNLABEL   : $UNLABEL_CSV"
echo "OUT ROOT  : $OUT_ROOT"
echo "==================================="

srun python ${PROJECT_ROOT}/src/build_npy_dataset.py \
    --csv ${EVENT_CSV} \
    --csv ${NOISE_CSV} \
    --csv ${UNLABEL_CSV} \
    --out_root ${OUT_ROOT} \
    --dtype float32 \
    --target_fs 1000 \
    --site_fs pohang:1000 \
    --site_fs utah:2000