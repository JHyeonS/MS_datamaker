#!/bin/bash
set -euo pipefail

ROOT="/home/ted1204/MS_datamaker"
cd "$ROOT"

mkdir -p scripts/cpu/legacy/{MS_datamaker_PH,MS_datamaker_UTAH_2019,MS_datamaker_UTAH_2023,final}
mkdir -p docs/reference/{pohang,utah2019,utah2023,final}
mkdir -p scripts/cpu/sbatch/{pohang,utah2019,utah2023,final}

# 1) legacy shell 이동
if [ -d scripts/cpu/MS_datamaker_PH ]; then
  mv scripts/cpu/MS_datamaker_PH/* scripts/cpu/legacy/MS_datamaker_PH/ 2>/dev/null || true
  rmdir scripts/cpu/MS_datamaker_PH 2>/dev/null || true
fi

if [ -d scripts/cpu/MS_datamaker_UTAH_2019 ]; then
  mv scripts/cpu/MS_datamaker_UTAH_2019/* scripts/cpu/legacy/MS_datamaker_UTAH_2019/ 2>/dev/null || true
  rmdir scripts/cpu/MS_datamaker_UTAH_2019 2>/dev/null || true
fi

if [ -d scripts/cpu/MS_datamaker_UTAH_2023 ]; then
  mv scripts/cpu/MS_datamaker_UTAH_2023/* scripts/cpu/legacy/MS_datamaker_UTAH_2023/ 2>/dev/null || true
  rmdir scripts/cpu/MS_datamaker_UTAH_2023 2>/dev/null || true
fi

if [ -d scripts/cpu/final ]; then
  mv scripts/cpu/final/* scripts/cpu/legacy/final/ 2>/dev/null || true
  rmdir scripts/cpu/final/old 2>/dev/null || true
  rmdir scripts/cpu/final 2>/dev/null || true
fi

# 2) src 안의 old/archive/reference 이동
mkdir -p docs/reference/pohang docs/reference/utah2019 docs/reference/utah2023 docs/reference/final

if [ -d src/datamaker/pohang/old ]; then
  mv src/datamaker/pohang/old docs/reference/pohang/
fi
if [ -d src/datamaker/pohang/archive ]; then
  mv src/datamaker/pohang/archive docs/reference/pohang/
fi
if [ -f src/datamaker/pohang/process.md ]; then
  mv src/datamaker/pohang/process.md docs/reference/pohang/process.md
fi

if [ -d src/datamaker/utah2019/old_image ]; then
  mv src/datamaker/utah2019/old_image docs/reference/utah2019/
fi

if [ -d src/datamaker/utah2023/old ]; then
  mv src/datamaker/utah2023/old docs/reference/utah2023/
fi

if [ -d src/datamaker/final/old ]; then
  mv src/datamaker/final/old docs/reference/final/
fi
if [ -f src/datamaker/final/run_experiments_splits.sh ]; then
  mv src/datamaker/final/run_experiments_splits.sh scripts/cpu/legacy/final/
fi
if [ -f src/datamaker/final/run_prepare_final_inputs.sh ]; then
  mv src/datamaker/final/run_prepare_final_inputs.sh scripts/cpu/legacy/final/
fi

# 3) 잘못 들어간 config 정리
if [ -f configs/datamaker/cpu_server.yaml ]; then
  rm -f configs/datamaker/cpu_server.yaml
fi

echo "[DONE] repo layout cleaned"