#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safe 1st-stage CPU repo restructure for MS_datamaker.

Usage:
    python tools/restructure_cpu_repo.py --repo_root /home/ted1204/MS_datamaker --dry_run
    python tools/restructure_cpu_repo.py --repo_root /home/ted1204/MS_datamaker

Behavior:
- Creates new Codex-friendly directory layout
- Copies legacy site/final script files into new structure
- Does NOT delete legacy folders
- Writes starter files if missing
- Tries to infer common legacy folder names
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, required=True, help="Path to MS_datamaker repo root")
    parser.add_argument("--dry_run", action="store_true", help="Print actions only")
    return parser.parse_args()


def safe_mkdir(path: Path, dry_run: bool):
    if not path.exists():
        print(f"[MKDIR] {path}")
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)


def safe_copy_file(src: Path, dst: Path, dry_run: bool):
    if not src.exists():
        print(f"[SKIP] source not found: {src}")
        return

    try:
        if src.resolve() == dst.resolve():
            print(f"[SKIP] same file: {src}")
            return
    except Exception:
        pass

    if dst.exists():
        try:
            if dst.resolve() == src.resolve():
                print(f"[SKIP] same file: {src}")
                return
        except Exception:
            pass
        print(f"[WARN] destination exists, overwrite: {dst}")

    print(f"[COPY] {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def safe_write_text(path: Path, content: str, dry_run: bool):
    if path.exists():
        print(f"[SKIP] file exists: {path}")
        return
    print(f"[WRITE] {path}")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def find_first_existing(root: Path, candidates: Iterable[str]) -> Path | None:
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def copy_scripts_from_dir(src_dir: Path | None, dst_dir: Path, dry_run: bool):
    if src_dir is None:
        print(f"[SKIP] no source dir for -> {dst_dir}")
        return
    print(f"[INFO] scanning source dir: {src_dir}")
    for path in src_dir.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml", ".md", ".txt"}:
            rel = path.relative_to(src_dir)
            safe_copy_file(path, dst_dir / rel, dry_run)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    dry_run = args.dry_run

    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")

    print("=" * 80)
    print(f"[INFO] repo_root: {repo_root}")
    print(f"[INFO] dry_run  : {dry_run}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Create target directory layout
    # ------------------------------------------------------------------
    target_dirs = [
        repo_root / "src" / "datamaker" / "common",
        repo_root / "src" / "datamaker" / "pohang",
        repo_root / "src" / "datamaker" / "utah2019",
        repo_root / "src" / "datamaker" / "utah2023",
        repo_root / "src" / "datamaker" / "final",
        repo_root / "src" / "shared",
        repo_root / "scripts" / "cpu",
        repo_root / "configs" / "datamaker",
        repo_root / "configs" / "experiments",
        repo_root / "configs" / "system",
        repo_root / "env",
        repo_root / "docs",
        repo_root / "tools",
    ]
    for d in target_dirs:
        safe_mkdir(d, dry_run)

    # package markers
    init_files = [
        repo_root / "src" / "__init__.py",
        repo_root / "src" / "datamaker" / "__init__.py",
        repo_root / "src" / "datamaker" / "common" / "__init__.py",
        repo_root / "src" / "datamaker" / "pohang" / "__init__.py",
        repo_root / "src" / "datamaker" / "utah2019" / "__init__.py",
        repo_root / "src" / "datamaker" / "utah2023" / "__init__.py",
        repo_root / "src" / "datamaker" / "final" / "__init__.py",
        repo_root / "src" / "shared" / "__init__.py",
    ]
    for f in init_files:
        safe_write_text(f, "", dry_run)

    # ------------------------------------------------------------------
    # 2. Try to locate legacy roots
    # ------------------------------------------------------------------
    pohang_src = find_first_existing(repo_root, [
        "MS_datamaker_PH/scripts",
        "MS_datamaker_PH/src",
        "MS_datamaker_PH",
        "datamaker_ph/scripts",
        "pohang/scripts",
    ])
    utah2019_src = find_first_existing(repo_root, [
        "MS_datamaker_UTAH_2019/scripts",
        "MS_datamaker_UTAH_2019/src",
        "MS_datamaker_UTAH_2019",
        "utah2019/scripts",
    ])
    utah2023_src = find_first_existing(repo_root, [
        "MS_datamaker_UTAH_2023/scripts",
        "MS_datamaker_UTAH_2023/src",
        "MS_datamaker_UTAH_2023",
        "utah2023/scripts",
    ])
    final_src = find_first_existing(repo_root, [
        "MS_datamaker_FINAL/scripts",
        "MS_datamaker_FINAL/src",
        "MS_datamaker_FINAL",
        "final/scripts",
    ])

    # ------------------------------------------------------------------
    # 3. Copy legacy python/shell/config/docs into new src layout
    # ------------------------------------------------------------------
    copy_scripts_from_dir(pohang_src, repo_root / "src" / "datamaker" / "pohang", dry_run)
    copy_scripts_from_dir(utah2019_src, repo_root / "src" / "datamaker" / "utah2019", dry_run)
    copy_scripts_from_dir(utah2023_src, repo_root / "src" / "datamaker" / "utah2023", dry_run)
    copy_scripts_from_dir(final_src, repo_root / "src" / "datamaker" / "final", dry_run)

    # ------------------------------------------------------------------
    # 4. Copy shell launchers into scripts/cpu from legacy launcher dirs
    #    Keep per-source subdirectories to avoid name collisions.
    # ------------------------------------------------------------------
    launcher_candidates = [
        repo_root / "MS_datamaker_PH" / "slurm",
        repo_root / "MS_datamaker_UTAH_2019" / "slurm",
        repo_root / "MS_datamaker_UTAH_2023" / "slurm",
        repo_root / "MS_datamaker_FINAL" / "slurm",
        repo_root / "slurm",
    ]

    for cand in launcher_candidates:
        if not cand.exists():
            continue

        source_tag = cand.parent.name

        for path in cand.rglob("*"):
            if not path.is_file() or path.suffix != ".sh":
                continue

            rel = path.relative_to(cand)
            dst = repo_root / "scripts" / "cpu" / source_tag / rel
            safe_copy_file(path, dst, dry_run)

    # ------------------------------------------------------------------
    # 5. Copy yaml configs from legacy roots into configs/datamaker
    # ------------------------------------------------------------------
    config_candidates = [
        repo_root / "MS_datamaker_PH",
        repo_root / "MS_datamaker_UTAH_2019",
        repo_root / "MS_datamaker_UTAH_2023",
        repo_root / "MS_datamaker_FINAL",
        repo_root / "config",
        repo_root / "configs",
    ]
    for cand in config_candidates:
        if cand.exists():
            for path in cand.rglob("*"):
                if path.is_file() and path.suffix in {".yaml", ".yml"}:
                    safe_copy_file(path, repo_root / "configs" / "datamaker" / path.name, dry_run)

    # ------------------------------------------------------------------
    # 6. Starter files
    # ------------------------------------------------------------------
    gitignore_text = """__pycache__/
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

output/
output_npy/
outputs/
old_outputs/
raw_data/
inventories/
segment_plans/
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
"""

    agents_text = """# AGENTS.md

## Project structure
- src/datamaker: dataset building pipeline
- src/shared: repo-wide shared utilities
- scripts/cpu: CPU-side entry scripts
- configs/datamaker: datamaker configs
- configs/system: machine-specific path configs

## Rules
- Do not commit raw_data/, output/, output_npy/, logs/, large generated artifacts
- Prefer config-driven paths over hardcoded absolute paths
- New CPU launchers go into scripts/cpu
- Site-specific code goes into src/datamaker/pohang, utah2019, utah2023
- Integrated multi-site pipeline goes into src/datamaker/final
- Use executable Python scripts when possible
"""

    cpu_system_yaml = """paths:
  repo_root: /home/ted1204/MS_datamaker
  raw_root: /home/ted1204/ms_datamaker_workspace/raw_data
  inventory_root: /home/ted1204/ms_datamaker_workspace/inventories
  segment_plan_root: /home/ted1204/ms_datamaker_workspace/segment_plans
  output_root: /home/ted1204/ms_datamaker_workspace/output
  output_npy_root: /home/ted1204/ms_datamaker_workspace/output_npy
  log_root: /home/ted1204/ms_datamaker_workspace/logs
"""

    env_cpu_yaml = """name: ms_datamaker
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
"""

    readme_text = """# MS_datamaker

CPU-side dataset builder for DAS microseismic detection.

## Layout
- src/datamaker/pohang: Pohang-specific preprocessing and plan generation
- src/datamaker/utah2019: Utah 2019-specific preprocessing and plan generation
- src/datamaker/utah2023: Utah 2023-specific preprocessing and plan generation
- src/datamaker/final: merged multi-site pipeline
- scripts/cpu: launcher scripts
- configs/datamaker: datamaker configs
- configs/system: system path configs

## Notes
Generated outputs should live outside the repo.
"""

    safe_write_text(repo_root / ".gitignore", gitignore_text, dry_run)
    safe_write_text(repo_root / "AGENTS.md", agents_text, dry_run)
    safe_write_text(repo_root / "configs" / "system" / "cpu_server.yaml", cpu_system_yaml, dry_run)
    safe_write_text(repo_root / "env" / "environment_cpu.yml", env_cpu_yaml, dry_run)
    safe_write_text(repo_root / "README.md", readme_text, dry_run)

    print("=" * 80)
    print("[DONE] CPU repo restructure scaffold complete")
    print("=" * 80)
    print("[NEXT]")
    print("1. Inspect src/datamaker/* for copied scripts")
    print("2. Inspect scripts/cpu for launcher duplication")
    print("3. Decide which legacy folders can remain as reference")
    print("4. Update launcher calls to use: python -m src.datamaker....")
    print("5. Run small tests before deleting any legacy dirs")


if __name__ == "__main__":
    main()