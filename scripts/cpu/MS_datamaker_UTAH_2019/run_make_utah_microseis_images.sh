#!/bin/bash
#SBATCH --job-name=utah_upper_lower
#SBATCH --output=/home/ted1204/MS_datamaker_UTAH/logs/utah_upper_lower_%j.out
#SBATCH --error=/home/ted1204/MS_datamaker_UTAH/logs/utah_upper_lower_%j.err
#SBATCH --partition=v3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00

set -e

cd /home/ted1204/MS_datamaker_UTAH

srun python scripts/make_utah_all_1s_images_upper_lower.py \
  --data_dir data/microseismic \
  --out_dir outputs/utah_1s_images_all \
  --fs 1000 \
  --win_sec 1.0 \
  --upper_ch_start 273 \
  --upper_ch_end 678 \
  --lower_ch_start 679 \
  --lower_ch_end 1084 \
  --fmin 5 \
  --fmax 80 \
  --cmap gray \
  --clip 3.0