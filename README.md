# MS Datamaker

Code for constructing DAS microseismic datasets used in cross-site transfer-learning experiments.

This repository contains the **processing code only**. Raw DAS data, processed NPY datasets, logs, figures, and restricted metadata are intentionally excluded from git.

## What This Repository Does

- Build site-specific inventories from DAS raw files
- Generate fixed time-channel window segment plans
- Merge Pohang, Utah 2019, and Utah 2023 metadata
- Apply preprocessing such as low-pass filtering, log-envelope scaling, and RMS normalization
- Create site-wise train/validation/test/pretraining splits
- Export PSD/frequency-analysis summaries and preview figures

## Directory Overview

```text
src/datamaker/
  pohang/        Pohang inventory and segment-plan utilities
  utah2019/      Utah 2019 inventory and segment-plan utilities
  utah2023/      Utah 2023 inventory and label-table utilities
  final/         Multi-site merge, preprocessing, analysis, and split generation

scripts/
  cpu/sbatch/    Slurm launch scripts for CPU preprocessing jobs
  data/          Dataset conversion utilities

configs/         Configuration templates
env/             Conda environment files
docs/            Method notes and setup documentation
tools/           Repository maintenance helpers
```

Legacy site folders are retained for reference:

```text
MS_datamaker_PH/
MS_datamaker_UTAH_2019/
MS_datamaker_UTAH_2023/
MS_datamaker_FINAL/
```

Production code should preferably live under `src/datamaker/`.

## Data Policy

The following are **not included** in this public repository:

- Raw DAS files (`.tdms`, `.sgy`, `.segy`)
- Generated `.npy` / `.npz` datasets
- Full metadata CSVs derived from restricted datasets
- KIGAM Pohang data
- Slurm logs and intermediate outputs
- Generated figures and presentation files

The `.gitignore` is intentionally strict to prevent accidental release of restricted or large files.

## Environment

Create the CPU environment:

```bash
conda env create -f env/environment_cpu.yml
conda activate ms_datamaker
```

On the current server, the existing environment may be named:

```bash
conda activate MS_datamaker
```

Set `PYTHONPATH` from the repository root:

```bash
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

## Basic Usage

Run module help from the repository root:

```bash
python -m datamaker.final.build_npy_dataset --help
python -m datamaker.final.make_ratio_balanced_splits_3site --help
python -m datamaker.final.analyze_raw_frequency_energy --help
```

Example Slurm launchers are under:

```text
scripts/cpu/sbatch/
```

These scripts assume that raw data and intermediate segment plans exist in local, ignored directories.

## Reproducibility Notes

The expected high-level pipeline is:

1. Build site inventories.
2. Generate event/noise/unlabeled segment plans.
3. Merge site-level segment plans.
4. Build preprocessed NPY windows.
5. Generate ratio-balanced experimental splits.
6. Run PSD/frequency analysis and visual checks.

Because the raw data are not distributed here, users must obtain the corresponding DAS datasets separately and place them in the expected local data directories.

## Data Availability Statement Template

```text
The code used for DAS microseismic dataset construction, preprocessing,
frequency analysis, and experimental split generation is available in this
repository. Raw Pohang DAS data are not publicly available due to data-use
restrictions. Utah FORGE DAS data should be obtained from the corresponding
Geothermal Data Repository project records. Processed datasets and generated
NPY files are not included due to size and licensing restrictions.
```

## License

Add a license before public release. For academic code release, MIT or BSD-3-Clause is usually appropriate, but confirm with all collaborators and data owners first.
