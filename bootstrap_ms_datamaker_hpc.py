#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import stat
from pathlib import Path
from textwrap import dedent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=str, required=True, help="New standardized project root")
    p.add_argument("--dry-run", action="store_true", help="Print actions only")
    p.add_argument("--force", action="store_true", help="Allow scaffolding into a non-empty directory")
    return p.parse_args()


def write_text(path: Path, content: str, dry_run: bool):
    print(f"[WRITE] {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_executable(path: Path, content: str, dry_run: bool):
    write_text(path, content, dry_run)
    if dry_run:
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def touch(path: Path, dry_run: bool):
    print(f"[TOUCH] {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def ensure_dir(path: Path, dry_run: bool):
    print(f"[MKDIR] {path}")
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def module_stub(doc: str) -> str:
    return dedent(f'''    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    {doc}
    """

    from __future__ import annotations

    import argparse


    def parse_args():
        parser = argparse.ArgumentParser()
        return parser.parse_args()


    def main():
        args = parse_args()
        print("TODO: replace stub with real implementation")
        print("module =", __file__)
        print("args   =", args)


    if __name__ == "__main__":
        main()
    ''')


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    if project_root.exists():
        existing = list(project_root.iterdir())
        if existing and not args.force:
            raise SystemExit(
                f"[ERROR] Project root is not empty: {project_root}\n"
                "Use --force to scaffold into a non-empty directory."
            )

    print("=" * 88)
    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] dry_run     : {args.dry_run}")
    print(f"[INFO] force       : {args.force}")
    print("=" * 88)

    dirs = [
        project_root / "src" / "datamaker" / "common",
        project_root / "src" / "datamaker" / "pohang",
        project_root / "src" / "datamaker" / "utah2019",
        project_root / "src" / "datamaker" / "utah2023",
        project_root / "src" / "datamaker" / "final",
        project_root / "src" / "shared",
        project_root / "scripts" / "cpu" / "sbatch" / "pohang",
        project_root / "scripts" / "cpu" / "sbatch" / "utah2019",
        project_root / "scripts" / "cpu" / "sbatch" / "utah2023",
        project_root / "scripts" / "cpu" / "sbatch" / "final",
        project_root / "scripts" / "cpu" / "legacy",
        project_root / "configs" / "system",
        project_root / "docs",
        project_root / "env",
        project_root / "logs",
        project_root / "tools",
    ]
    for d in dirs:
        ensure_dir(d, args.dry_run)

    init_files = [
        project_root / "src" / "__init__.py",
        project_root / "src" / "datamaker" / "__init__.py",
        project_root / "src" / "datamaker" / "common" / "__init__.py",
        project_root / "src" / "datamaker" / "pohang" / "__init__.py",
        project_root / "src" / "datamaker" / "utah2019" / "__init__.py",
        project_root / "src" / "datamaker" / "utah2023" / "__init__.py",
        project_root / "src" / "datamaker" / "final" / "__init__.py",
        project_root / "src" / "shared" / "__init__.py",
    ]
    for f in init_files:
        touch(f, args.dry_run)

    gitignore = dedent('''    __pycache__/
    *.pyc
    *.pyo
    *.pyd

    .venv/
    venv/
    env/

    .ipynb_checkpoints/

    .DS_Store
    Thumbs.db

    logs/
    *.log
    *.err
    *.out

    raw_data/
    inventories/
    segment_plans/
    outputs/
    output/
    output_npy/
    old_outputs/
    datasets/
    data/

    *.npy
    *.npz
    *.h5
    *.hdf5
    *.tdms
    *.sgy
    *.segy
    *.gz
    *.[0-9]

    *.png
    *.jpg
    *.jpeg
    ''')

    agents = dedent('''    # AGENTS.md

    ## Core philosophy
    - Site-specific acquisition and preprocessing remain independent.
    - Each site produces intermediate artifacts in its own way.
    - The FINAL layer consumes site outputs and creates a unified training-ready dataset.

    ## Project structure
    - src/datamaker/pohang: Pohang-specific preprocessing
    - src/datamaker/utah2019: Utah 2019-specific preprocessing
    - src/datamaker/utah2023: Utah 2023-specific preprocessing
    - src/datamaker/final: integration layer
    - scripts/cpu/sbatch: SLURM launchers
    - scripts/cpu/legacy: legacy copied shell scripts for reference only

    ## Rules
    - Do not commit generated datasets or large raw files.
    - Do not force artificial uniformity across site-specific raw-data logic.
    - Standardize execution through sbatch launchers and Python module entry points.
    - FINAL integrates site outputs; it does not replace site-specific preprocessing.
    ''')

    readme = dedent('''    # MS_datamaker (HPC scaffold)

    This scaffold reflects a site-aware DAS microseismic dataset-building workflow.

    ## Design philosophy
    1. Each site is processed independently because raw data structure, labels, and metadata differ.
    2. Site pipelines export intermediate artifacts such as inventories and segment plans.
    3. The FINAL layer integrates site outputs into a unified training-ready dataset.
    4. On HPC, execution is driven by SLURM sbatch launchers and dependency chaining.

    ## Suggested submission flow
    ```bash
    sbatch scripts/cpu/sbatch/pohang/prepare.sh
    sbatch scripts/cpu/sbatch/utah2019/prepare.sh
    sbatch scripts/cpu/sbatch/utah2023/prepare.sh
    sbatch scripts/cpu/sbatch/final/full_pipeline.sh
    ```

    Or:
    ```bash
    bash scripts/cpu/submit_all.sh
    ```
    ''')

    cpu_system = dedent('''    paths:
      project_root: /home/ted1204/MS_datamaker_std
      raw_root: /home/ted1204/ms_datamaker_workspace/raw_data
      inventory_root: /home/ted1204/ms_datamaker_workspace/inventories
      segment_plan_root: /home/ted1204/ms_datamaker_workspace/segment_plans
      output_root: /home/ted1204/ms_datamaker_workspace/output
      output_npy_root: /home/ted1204/ms_datamaker_workspace/output_npy
      logs_root: /home/ted1204/ms_datamaker_workspace/logs
    ''')

    env_cpu = dedent('''    name: ms_datamaker
    channels:
      - conda-forge
    dependencies:
      - python=3.10
      - numpy
      - scipy
      - pandas
      - matplotlib
      - scikit-learn
      - pyyaml
      - tqdm
      - pip
    ''')

    write_text(project_root / ".gitignore", gitignore, args.dry_run)
    write_text(project_root / "AGENTS.md", agents, args.dry_run)
    write_text(project_root / "README.md", readme, args.dry_run)
    write_text(project_root / "configs" / "system" / "cpu_server.yaml", cpu_system, args.dry_run)
    write_text(project_root / "env" / "environment_cpu.yml", env_cpu, args.dry_run)

    starter_modules = {
        project_root / "src" / "datamaker" / "pohang" / "make_file_inventory.py":
            module_stub("Pohang-specific inventory builder."),
        project_root / "src" / "datamaker" / "pohang" / "make_segment_plan_event_pohang.py":
            module_stub("Pohang-specific event segment plan builder."),
        project_root / "src" / "datamaker" / "pohang" / "make_segment_plan_noise_unlabel_pohang.py":
            module_stub("Pohang-specific noise/unlabel segment plan builder."),
        project_root / "src" / "datamaker" / "utah2019" / "make_inventory_utah2019.py":
            module_stub("Utah 2019-specific inventory builder."),
        project_root / "src" / "datamaker" / "utah2019" / "make_segment_plan_event_utah2019.py":
            module_stub("Utah 2019-specific event segment plan builder."),
        project_root / "src" / "datamaker" / "utah2019" / "make_segment_plan_noise_unlabel_utah2019.py":
            module_stub("Utah 2019-specific noise/unlabel segment plan builder."),
        project_root / "src" / "datamaker" / "utah2023" / "merge_metadata_and_labels.py":
            module_stub("Utah 2023-specific metadata/label merger."),
        project_root / "src" / "datamaker" / "utah2023" / "make_inventory_utah2023.py":
            module_stub("Utah 2023-specific inventory builder."),
        project_root / "src" / "datamaker" / "utah2023" / "make_segment_plan_from_labels_utah2023.py":
            module_stub("Utah 2023-specific label-driven segment plan builder."),
        project_root / "src" / "datamaker" / "utah2023" / "sgy_to_npy_dataset.py":
            module_stub("Utah 2023-specific SGY to NPY converter."),
        project_root / "src" / "datamaker" / "final" / "merge_inventories.py":
            module_stub("Merge site-specific inventories into an integration-layer inventory."),
        project_root / "src" / "datamaker" / "final" / "merge_segment_plans.py":
            module_stub("Merge site-specific segment plans into a unified plan."),
        project_root / "src" / "datamaker" / "final" / "validate_merged_plans.py":
            module_stub("Validate merged inventories and segment plans before dataset build."),
        project_root / "src" / "datamaker" / "final" / "build_npy_dataset.py":
            module_stub("Build final NPY dataset from merged plans."),
        project_root / "src" / "datamaker" / "final" / "make_experiment_splits_3site.py":
            module_stub("Create final 3-site experiment splits from merged dataset metadata."),
    }
    for path, content in starter_modules.items():
        write_text(path, content, args.dry_run)

    sbatch_templates = {
        project_root / "scripts" / "cpu" / "sbatch" / "pohang" / "prepare.sh": dedent('''            #!/bin/bash
            #SBATCH --job-name=ph_prepare
            #SBATCH --output=logs/ph_prepare_%j.out
            #SBATCH --error=logs/ph_prepare_%j.err
            #SBATCH --partition=v3
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=4
            #SBATCH --time=24:00:00
            #SBATCH --mem=16G

            set -euo pipefail
            eval "$(conda shell.bash hook)"
            conda activate ms_datamaker

            PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
            cd "${PROJECT_ROOT}"
            export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

            python -m src.datamaker.pohang.make_file_inventory
            python -m src.datamaker.pohang.make_segment_plan_event_pohang
            python -m src.datamaker.pohang.make_segment_plan_noise_unlabel_pohang
            '''),
        project_root / "scripts" / "cpu" / "sbatch" / "utah2019" / "prepare.sh": dedent('''            #!/bin/bash
            #SBATCH --job-name=ut19_prepare
            #SBATCH --output=logs/ut19_prepare_%j.out
            #SBATCH --error=logs/ut19_prepare_%j.err
            #SBATCH --partition=v3
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=4
            #SBATCH --time=24:00:00
            #SBATCH --mem=16G

            set -euo pipefail
            eval "$(conda shell.bash hook)"
            conda activate ms_datamaker

            PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
            cd "${PROJECT_ROOT}"
            export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

            python -m src.datamaker.utah2019.make_inventory_utah2019
            python -m src.datamaker.utah2019.make_segment_plan_event_utah2019
            python -m src.datamaker.utah2019.make_segment_plan_noise_unlabel_utah2019
            '''),
        project_root / "scripts" / "cpu" / "sbatch" / "utah2023" / "prepare.sh": dedent('''            #!/bin/bash
            #SBATCH --job-name=ut23_prepare
            #SBATCH --output=logs/ut23_prepare_%j.out
            #SBATCH --error=logs/ut23_prepare_%j.err
            #SBATCH --partition=v3
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=4
            #SBATCH --time=24:00:00
            #SBATCH --mem=16G

            set -euo pipefail
            eval "$(conda shell.bash hook)"
            conda activate ms_datamaker

            PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
            cd "${PROJECT_ROOT}"
            export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

            python -m src.datamaker.utah2023.merge_metadata_and_labels
            python -m src.datamaker.utah2023.make_inventory_utah2023
            python -m src.datamaker.utah2023.make_segment_plan_from_labels_utah2023
            python -m src.datamaker.utah2023.sgy_to_npy_dataset
            '''),
        project_root / "scripts" / "cpu" / "sbatch" / "final" / "full_pipeline.sh": dedent('''            #!/bin/bash
            #SBATCH --job-name=final_pipeline
            #SBATCH --output=logs/final_pipeline_%j.out
            #SBATCH --error=logs/final_pipeline_%j.err
            #SBATCH --partition=v3
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=8
            #SBATCH --time=48:00:00
            #SBATCH --mem=32G

            set -euo pipefail
            eval "$(conda shell.bash hook)"
            conda activate ms_datamaker

            PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
            cd "${PROJECT_ROOT}"
            export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

            python -m src.datamaker.final.merge_inventories
            python -m src.datamaker.final.merge_segment_plans
            python -m src.datamaker.final.validate_merged_plans
            python -m src.datamaker.final.build_npy_dataset
            python -m src.datamaker.final.make_experiment_splits_3site
            '''),
        project_root / "scripts" / "cpu" / "submit_all.sh": dedent('''            #!/bin/bash
            set -euo pipefail

            PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
            cd "${PROJECT_ROOT}"

            jid_pohang=$(sbatch --parsable scripts/cpu/sbatch/pohang/prepare.sh)
            echo "[SUBMIT] pohang   : ${jid_pohang}"

            jid_ut19=$(sbatch --parsable scripts/cpu/sbatch/utah2019/prepare.sh)
            echo "[SUBMIT] utah2019 : ${jid_ut19}"

            jid_ut23=$(sbatch --parsable scripts/cpu/sbatch/utah2023/prepare.sh)
            echo "[SUBMIT] utah2023 : ${jid_ut23}"

            dep="${jid_pohang}:${jid_ut19}:${jid_ut23}"
            jid_final=$(sbatch --parsable --dependency=afterok:${dep} scripts/cpu/sbatch/final/full_pipeline.sh)
            echo "[SUBMIT] final    : ${jid_final}"
            '''),
    }
    for path, content in sbatch_templates.items():
        write_executable(path, content, args.dry_run)

    migration = dedent('''    # Migration notes

    Keep the original folders as reference:
    - MS_datamaker_PH
    - MS_datamaker_UTAH_2019
    - MS_datamaker_UTAH_2023
    - MS_datamaker_FINAL

    Suggested flow:
    1. Copy proven site-specific Python scripts into src/datamaker/{site}/
    2. Move old shell scripts into scripts/cpu/legacy/
    3. Replace placeholder module bodies with real code
    4. Adjust sbatch resources and arguments
    5. Test site pipelines separately
    6. Then test FINAL integration
    7. Finally use submit_all.sh
    ''')
    write_text(project_root / "docs" / "migration_notes.md", migration, args.dry_run)

    print("=" * 88)
    print("[DONE] HPC scaffold created")
    print("=" * 88)
    print("[NEXT]")
    print("1. Copy proven site-specific code into src/datamaker/{site}/")
    print("2. Move old shell scripts into scripts/cpu/legacy/")
    print("3. Replace stubs with real code")
    print("4. Fill actual arguments/paths in sbatch launchers")
    print("5. Test site jobs first, then FINAL, then submit_all.sh")


if __name__ == "__main__":
    main()
