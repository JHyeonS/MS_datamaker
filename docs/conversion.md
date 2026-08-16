# MS_datamaker Conversion Guide

This repository is in the second stage of migration.

The old site folders are kept as references:

- `MS_datamaker_PH/`
- `MS_datamaker_UTAH_2019/`
- `MS_datamaker_UTAH_2023/`
- `MS_datamaker_FINAL/`

The active workflow should use the package layout:

- `src/datamaker/pohang/`
- `src/datamaker/utah2019/`
- `src/datamaker/utah2023/`
- `src/datamaker/final/`

## Rule

Run scripts from the repository root with `src` on `PYTHONPATH`:

```bash
cd /home/ted1204/MS_datamaker
PYTHONPATH=/home/ted1204/MS_datamaker/src python -m datamaker.pohang.make_file_inventory --help
```

Do not add new production entry points under the legacy folders.

## Canonical Pipeline

1. Pohang preparation

```bash
sbatch scripts/cpu/sbatch/pohang/prepare.sh
```

2. Utah 2019 preparation

```bash
sbatch scripts/cpu/sbatch/utah2019/prepare.sh
```

3. Utah 2023 preparation

```bash
sbatch scripts/cpu/sbatch/utah2023/prepare.sh
```

4. Final merge, NPY build, and experiment splits

```bash
sbatch scripts/cpu/sbatch/final/full_pipeline.sh
```

Or submit the dependency chain:

```bash
bash scripts/cpu/submit_all.sh
```

## Data Contract

Each site-specific preparation stage should produce these files under `outputs/<site>/`:

- `inventory.csv`
- `segment_plan_event.csv`
- `segment_plan_noise.csv`
- `segment_plan_unlabel.csv`

Utah 2023 now follows the same convention and produces event/noise/unlabel plan files.

The final stage consumes those plan files and writes:

- `outputs/final/inventory_all.csv`
- `outputs/final/segment_plan_event.csv`
- `outputs/final/segment_plan_noise.csv`
- `outputs/final/segment_plan_unlabel.csv`
- `outputs/final/segment_plan_all.csv`
- `outputs/final/experiments/`
- `output_npy/final/`

## Migration Checklist

- Keep legacy folders until the new launchers reproduce the expected outputs.
- Put machine-specific paths in `configs/system/cpu_server.yaml`.
- Keep generated data, logs, NPY files, SEG-Y files, and TDMS files out of git.
- Add new reusable logic under `src/datamaker/common/`.
- Add new site-specific logic only under the matching `src/datamaker/<site>/` package.
- Once the new workflow is verified, move legacy folders under `legacy/` with `git mv`.

## Verification Commands

These commands check that the active module paths resolve:

```bash
PYTHONPATH=/home/ted1204/MS_datamaker/src python -m datamaker.pohang.make_file_inventory --help
PYTHONPATH=/home/ted1204/MS_datamaker/src python -m datamaker.utah2019.make_inventory_utah2019 --help
PYTHONPATH=/home/ted1204/MS_datamaker/src python -m datamaker.utah2023.make_inventory_utah2023 --help
PYTHONPATH=/home/ted1204/MS_datamaker/src python -m datamaker.final.merge_inventories --help
```
