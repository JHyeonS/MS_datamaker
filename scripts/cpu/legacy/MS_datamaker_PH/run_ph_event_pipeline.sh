#!/bin/sh

#SBATCH --job-name=ph_event_pipeline
#SBATCH --output=ph_event_pipeline.out
#SBATCH --partition=v3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

cd /home/ted1204/MS_datamaker_PH

echo "STEP 2: overlap catalog 생성"
srun python scripts/make_overlap_catalog.py

echo "STEP 3: catalog - TDMS 매칭"
srun python scripts/match_catalog_tdms.py \
  --catalog_xlsx outputs/overlap_catalog.xlsx \
  --tdms_csv outputs/tdms_file_index.csv \
  --out_csv outputs/catalog_tdms_intersection.csv

echo "STEP 4: TDMS 복사"
srun python scripts/copy_matched_tdms.py \
  --match_csv outputs/catalog_tdms_intersection.csv \
  --dest_dir data/raw_tdms

echo "STEP 5: 10초 이미지 생성"
srun python scripts/make_tdms_10s_images.py \
  --match_csv outputs/catalog_tdms_intersection.csv \
  --out_dir outputs/tdms_10s_images_bp_robust_1s \
  --segment_sec 1 \
  --fs 1000 \
  --ch_start 243 \
  --ch_end 649 \
  --fmin 5 \
  --fmax 80 \
  --cmap gray \
  --clip 3.0

echo "DONE"

exit 0