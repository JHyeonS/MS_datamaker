#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_COLS = ["site", "label", "sample_index_within_class"]
SPLIT_NAMES = ["pretrain", "train", "val", "test"]


def make_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing key columns: {missing}")

    out = df.copy()
    out["site"] = out["site"].astype(str)
    out["label"] = out["label"].astype(int)
    out["sample_index_within_class"] = out["sample_index_within_class"].astype(int)
    return out


def build_lookup(current_all: pd.DataFrame) -> pd.DataFrame:
    current_all = make_key_frame(current_all)
    duplicated = current_all.duplicated(KEY_COLS, keep=False)
    if duplicated.any():
        examples = current_all.loc[duplicated, KEY_COLS + ["npy_path"]].head(10)
        raise ValueError(f"Current all_samples has duplicate mapping keys:\n{examples}")

    return current_all.set_index(KEY_COLS, drop=False)


def map_split_csv(old_csv: Path, lookup: pd.DataFrame, out_csv: Path) -> dict:
    old_df = make_key_frame(pd.read_csv(old_csv))
    old_df = old_df.reset_index(drop=True)

    mapped_rows = []
    missing_rows = []
    for i, row in old_df.iterrows():
        key = (str(row["site"]), int(row["label"]), int(row["sample_index_within_class"]))
        if key not in lookup.index:
            missing_rows.append({**row.to_dict(), "old_row_index": int(i)})
            continue
        mapped_rows.append(lookup.loc[key].to_dict())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mapped_df = pd.DataFrame(mapped_rows)
    mapped_df.to_csv(out_csv, index=False)

    missing_csv = None
    if missing_rows:
        missing_csv = out_csv.with_suffix(".missing.csv")
        pd.DataFrame(missing_rows).to_csv(missing_csv, index=False)

    return {
        "old_csv": str(old_csv),
        "out_csv": str(out_csv),
        "n_old_rows": int(len(old_df)),
        "n_mapped_rows": int(len(mapped_df)),
        "n_missing_rows": int(len(missing_rows)),
        "missing_csv": str(missing_csv) if missing_csv is not None else None,
        "site_counts": mapped_df["site"].value_counts().to_dict() if len(mapped_df) else {},
        "label_name_counts": mapped_df["label_name"].value_counts().to_dict()
        if len(mapped_df) and "label_name" in mapped_df.columns
        else {},
    }


def map_experiments(old_experiments: Path, current_all_csv: Path, out_dir: Path) -> dict:
    current_all = pd.read_csv(current_all_csv)
    lookup = build_lookup(current_all)

    report = {
        "old_experiments": str(old_experiments),
        "current_all_csv": str(current_all_csv),
        "out_dir": str(out_dir),
        "mapping_key": KEY_COLS,
        "stages": {},
    }

    for stage_dir in sorted(p for p in old_experiments.iterdir() if p.is_dir()):
        stage_report = {}
        out_stage = out_dir / stage_dir.name
        out_stage.mkdir(parents=True, exist_ok=True)

        old_summary = stage_dir / "summary.json"
        if old_summary.exists():
            (out_stage / "summary_0406_original.json").write_text(
                old_summary.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        for split in SPLIT_NAMES:
            old_csv = stage_dir / f"{split}.csv"
            if not old_csv.exists():
                continue
            stage_report[split] = map_split_csv(
                old_csv=old_csv,
                lookup=lookup,
                out_csv=out_stage / f"{split}.csv",
            )

        report["stages"][stage_dir.name] = stage_report

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "mapping_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map 0406 experiment splits to a current NPY dataset.")
    parser.add_argument("--old_experiments", required=True)
    parser.add_argument("--current_all_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = map_experiments(
        old_experiments=Path(args.old_experiments),
        current_all_csv=Path(args.current_all_csv),
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])
    print(f"[DONE] wrote: {Path(args.out_dir) / 'mapping_report.json'}")


if __name__ == "__main__":
    main()
