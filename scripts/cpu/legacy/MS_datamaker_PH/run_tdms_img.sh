#!/bin/sh

#SBATCH --job-name=tdms_img
#SBATCH --output=tdms_img.out
#SBATCH --partition=v3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

cd /home/ted1204/MS_datamaker_PH

srun python scripts/make_tdms_10s_images.py \
  --match_csv outputs/catalog_tdms_intersection.csv \
  --out_dir outputs/tdms_1s_images_raw \
  --segment_sec 1 \
  --fs 1000 \
  --max_events 200 \
  --ch_start 243 \
  --ch_end 649

exit 0