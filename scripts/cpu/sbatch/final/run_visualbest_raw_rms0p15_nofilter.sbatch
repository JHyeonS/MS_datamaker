#!/bin/bash
#SBATCH --job-name=raw_rms0p15
#SBATCH --output=logs/raw_rms0p15_%j.out
#SBATCH --error=logs/raw_rms0p15_%j.err
#SBATCH --partition=v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate MS_datamaker

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="visualbest_raw_rms_fs1000_rms0p15_nofilter"
NPY_ROOT="${PROJECT_ROOT}/output_npy/final/${RUN_NAME}"
PLAN_ROOT="${PROJECT_ROOT}/outputs/final_3site_input"

echo "[START] $(date)"
echo "[RUN_NAME] ${RUN_NAME}"

PREPROCESS_RUN_NAME="${RUN_NAME}" \
PREPROCESS="raw_rms" \
TARGET_FS="1000" \
TARGET_RMS="0.15" \
RMS_SCOPE="site" \
FILTER_SPEC_POHANG="none:none" \
FILTER_SPEC_UTAH2019="none:none" \
FILTER_SPEC_UTAH2023="none:none" \
PLAN_ROOT="${PLAN_ROOT}" \
OVERWRITE="1" \
bash scripts/cpu/sbatch/final/build_preprocess_from_final_3site_input.sh

python -m datamaker.final.make_ratio_balanced_splits_3site \
  --all_csv "${NPY_ROOT}/metadata/all_samples.csv" \
  --out_dir "${NPY_ROOT}/metadata/experiments_ratio_balanced" \
  --sites pohang utah_2019 utah_2023 \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
  --seed 42 --max_split_tries 1000 --max_unlabel_per_site 3000 \
  --min_train_noise 1 --min_train_event 1 \
  --min_val_noise 1 --min_val_event 1 \
  --min_test_noise 1 --min_test_event 1 \
  --min_val_event_utah2019 20 --min_test_event_utah2019 20

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -d "${NPY_ROOT}/metadata/experiments" ]]; then
  mv "${NPY_ROOT}/metadata/experiments" "${NPY_ROOT}/metadata/experiments_before_ratio_balanced_${STAMP}"
fi
cp -a "${NPY_ROOT}/metadata/experiments_ratio_balanced" "${NPY_ROOT}/metadata/experiments"

python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path('/home/ted1204/MS_datamaker/output_npy/final/visualbest_raw_rms_fs1000_rms0p15_nofilter')
print('[VERIFY]', root)
for exp in ['stage1_pohang_only','stage1_utah_2019_only','stage1_utah_2023_only','stage2_joint_all']:
    print('##', exp)
    for split in ['train','val','test','pretrain']:
        p = root/'metadata/experiments'/exp/f'{split}.csv'
        df = pd.read_csv(p)
        print(split, len(df), df['label'].value_counts().sort_index().to_dict())
PY

echo "[DONE] $(date)"
