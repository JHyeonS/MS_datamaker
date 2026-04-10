#!/bin/bash
#SBATCH --job-name=ph_test
#SBATCH --output=logs/ph_test_%j.out
#SBATCH --error=logs/ph_test_%j.err
#SBATCH --partition=v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate MS_datamaker

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[DEBUG] pwd=$(pwd)"
echo "[DEBUG] PYTHONPATH=${PYTHONPATH}"
python -m src.datamaker.pohang.make_file_inventory --help