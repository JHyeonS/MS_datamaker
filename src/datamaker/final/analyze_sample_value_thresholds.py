#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_token(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _resolve_output_dir(all_csv: Path, out_dir: str | None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    return all_csv.parent


def _summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=group_cols)

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            total_samples=("exceeds_abs_threshold", "size"),
            exceed_samples=("exceeds_abs_threshold", "sum"),
            positive_exceed_samples=("exceeds_positive_threshold", "sum"),
            negative_exceed_samples=("exceeds_negative_threshold", "sum"),
            max_abs=("sample_abs_max", "max"),
            max_value=("sample_max", "max"),
            min_value=("sample_min", "min"),
        )
        .reset_index()
    )
    grouped["exceed_fraction"] = (
        grouped["exceed_samples"] / grouped["total_samples"].clip(lower=1)
    )
    return grouped.sort_values(
        ["exceed_samples", "max_abs"], ascending=[False, False]
    ).reset_index(drop=True)


def analyze_thresholds(all_csv: Path, out_dir: Path, threshold: float) -> Dict:
    df = pd.read_csv(all_csv)
    required = {"npy_path", "site", "label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {all_csv}: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, row in df.iterrows():
        npy_path = Path(str(row["npy_path"]))
        arr = np.load(npy_path, mmap_mode="r")
        sample_min = float(np.min(arr))
        sample_max = float(np.max(arr))
        sample_abs_max = float(max(abs(sample_min), abs(sample_max)))

        exceeds_positive = sample_max > threshold
        exceeds_negative = sample_min < -threshold
        exceeds_abs = exceeds_positive or exceeds_negative

        rows.append(
            {
                "row_index": int(idx),
                "npy_path": str(npy_path),
                "site": str(row["site"]),
                "label": int(row["label"]),
                "label_name": str(row["label_name"]) if "label_name" in df.columns else "",
                "group_id": str(row["group_id"]) if "group_id" in df.columns else "",
                "sample_min": sample_min,
                "sample_max": sample_max,
                "sample_abs_max": sample_abs_max,
                "exceeds_positive_threshold": bool(exceeds_positive),
                "exceeds_negative_threshold": bool(exceeds_negative),
                "exceeds_abs_threshold": bool(exceeds_abs),
            }
        )

    stats = pd.DataFrame(rows)
    token = _safe_token(threshold)
    all_stats_path = out_dir / f"sample_value_stats_abs_gt_{token}.csv"
    exceed_path = out_dir / f"sample_value_exceedance_abs_gt_{token}.csv"
    by_site_label_path = out_dir / f"sample_value_exceedance_summary_by_site_label_abs_gt_{token}.csv"
    by_site_path = out_dir / f"sample_value_exceedance_summary_by_site_abs_gt_{token}.csv"
    summary_path = out_dir / f"sample_value_exceedance_summary_abs_gt_{token}.json"

    stats.to_csv(all_stats_path, index=False)
    stats[stats["exceeds_abs_threshold"]].sort_values(
        ["sample_abs_max", "site", "label_name"], ascending=[False, True, True]
    ).to_csv(exceed_path, index=False)

    by_site_label = _summarize_group(stats, ["site", "label", "label_name"])
    by_site = _summarize_group(stats, ["site"])
    by_site_label.to_csv(by_site_label_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    n_total = int(len(stats))
    n_exceed = int(stats["exceeds_abs_threshold"].sum())
    summary = {
        "all_csv": str(all_csv),
        "threshold": float(threshold),
        "n_total": n_total,
        "n_exceed_abs_threshold": n_exceed,
        "exceed_fraction": float(n_exceed / max(1, n_total)),
        "max_abs": float(stats["sample_abs_max"].max()) if n_total else None,
        "max_value": float(stats["sample_max"].max()) if n_total else None,
        "min_value": float(stats["sample_min"].min()) if n_total else None,
        "files": {
            "all_sample_stats": str(all_stats_path),
            "exceedance_samples": str(exceed_path),
            "summary_by_site_label": str(by_site_label_path),
            "summary_by_site": str(by_site_path),
            "summary_json": str(summary_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-sample min/max values after preprocessing."
    )
    parser.add_argument("--all_csv", required=True, help="metadata/all_samples.csv")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to all_csv parent.")
    parser.add_argument("--threshold", type=float, default=7.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_thresholds(
        all_csv=Path(args.all_csv),
        out_dir=_resolve_output_dir(Path(args.all_csv), args.out_dir),
        threshold=float(args.threshold),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
