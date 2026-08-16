#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"
mkdir -p logs

jid_pohang=$(sbatch --parsable scripts/cpu/sbatch/pohang/prepare.sh)
echo "[SUBMIT] pohang   : ${jid_pohang}"

jid_ut19=$(sbatch --parsable scripts/cpu/sbatch/utah2019/prepare.sh)
echo "[SUBMIT] utah2019 : ${jid_ut19}"

jid_ut23=$(sbatch --parsable scripts/cpu/sbatch/utah2023/prepare.sh)
echo "[SUBMIT] utah2023 : ${jid_ut23}"

dep="${jid_pohang}:${jid_ut19}:${jid_ut23}"
jid_final=$(sbatch --parsable --dependency=afterok:${dep} scripts/cpu/sbatch/final/full_pipeline.sh)
echo "[SUBMIT] final    : ${jid_final}"
