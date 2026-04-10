#!/bin/bash
#SBATCH --job-name=utah_splits
#SBATCH --output=/home/ted1204/MS_datamaker_UTAH_2023/logs/utah_splits_%j.out
#SBATCH --error=/home/ted1204/MS_datamaker_UTAH_2023/logs/utah_splits_%j.err
#SBATCH --partition=v3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00

set -e

cd /home/ted1204/MS_datamaker_UTAH_2023
mkdir -p ./logs

srun python -u ./scripts/make_utah_microseis_images_splits.py \
  --data_dir ./data/event_2417 \
  --out_dir ./outputs/utah_2023_2s_overlap_1s_second \
  --fs 1000 \
  --win_sec 2.0 \
  --stride_sec 1.0 \
  --splits "4000-4405/4450-4855/4900-5305/5350-5755" \
  --channel_map_csv ./FORGE_DAS_Circulation2023_ChannelMapping.csv \
  --input_channel_base 0 \
  --fmin 5 \
  --fmax 80 \
  --cmap gray \
  --clip 3.0