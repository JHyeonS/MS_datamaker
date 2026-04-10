#!/bin/bash
#SBATCH --job-name=utah_npy
#SBATCH --output=/home/ted1204/MS_datamaker_UTAH_2023/logs/utah_npy_%j.out
#SBATCH --error=/home/ted1204/MS_datamaker_UTAH_2023/logs/utah_npy_%j.err
#SBATCH --partition=v3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00

set -e

cd /home/ted1204/MS_datamaker_UTAH_2023
mkdir -p ./logs

srun python -u ./scripts/sgy_to_npy_dataset.py \
  --data_dir ./data/event_2417 \
  --out_dir ./outputs_npy/utah_2023_2s_6split \
  --site utah_2023 \
  --fs 1000 \
  --target_fs 1000 \
  --win_sec 2.0 \
  --stride_sec 1.0 \
  --splits "400-805/850-1255/1300-1705/1750-2155/2200-2605/2650-3055" \
  --input_channel_base 0 \
  --expected_channels 406 \
  --expected_samples 2000 \
  --dtype float32
