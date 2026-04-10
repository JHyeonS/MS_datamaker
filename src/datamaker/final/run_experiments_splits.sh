#!/bin/bash
set -euo pipefail

cd /home/ted1204/MS_Detection_Datamaker/MS_datamaker_FINAL

python /home/ted1204/MS_Detection_Datamaker/MS_datamaker_FINAL/scripts/make_experiment_splits_3site.py \
  --all_csv /home/ted1204/MS_Detection_Datamaker/MS_datamaker_FINAL/output_npy/metadata/all_samples.csv \
  --out_dir /home/ted1204/MS_Detection_Datamaker/MS_datamaker_FINAL/output_npy/metadata/experiments \
  --sites pohang utah_2019 utah_2023 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --max_split_tries 1000 \
  --max_unlabel_per_site 1500 \
  --min_train_noise 60 \
  --min_train_event 60 \
  --min_val_noise 20 \
  --min_val_event 20 \
  --min_test_noise 20 \
  --min_test_event 20 \
  --min_val_event_utah2019 15 \
  --min_test_event_utah2019 15