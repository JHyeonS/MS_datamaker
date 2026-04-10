#!/bin/sh

#SBATCH --job-name=make_inventory_pohang
#SBATCH --output=make_inventory_pohang.out
#SBATCH --partition=v3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate MS_datamaker

cd /home/ted1204/MS_Detection_Datamaker/MS_datamaker_PH

echo "Start make_inventory_pohang"
srun python scripts/make_file_inventory.py \
  --event_dir /home/ted1204/MS_Detection_Datamaker/MS_datamaker_PH/data/event \
  --noise_dir /home/ted1204/MS_Detection_Datamaker/MS_datamaker_PH/data/noise \
  --unlabel_dir /home/ted1204/MS_Detection_Datamaker/MS_datamaker_PH/data/unlabel \
  --out_csv /home/ted1204/MS_Detection_Datamaker/MS_datamaker_PH/outputs/inventory.csv \
  --site pohang \
  --view pohang \
  --original_fs 1000 \
  --ch_start 243 \
  --ch_end 648

echo "DONE"

exit 0