#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/ted1204/MS_datamaker"
cd "${PROJECT_ROOT}"

JOB_SCRIPT="scripts/cpu/sbatch/final/run_visualbest_raw_rms0p15_nofilter.sbatch"

sbatch "${JOB_SCRIPT}"
