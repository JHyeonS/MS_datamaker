#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from datamaker.final.build_npy_dataset import (
    LABEL_ID_TO_DIR,
    frequency_stats_to_frame,
    infer_label_from_row,
    read_raw_window,
    resolve_channel_range,
    resolve_file_path,
    resolve_original_fs,
    resolve_start_end_sec,
    sanitize_site_name,
    update_frequency_stats,
)


DEFAULT_PLAN_FILES = [
    "segment_plan_event.csv",
    "segment_plan_noise.csv",
    "segment_plan_unlabel.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze raw signal frequency energy from segment plans, before any "
            "dataset preprocessing/filter/RMS normalization."
        )
    )
    parser.add_argument(
        "--plan_dir",
        default="outputs/final_3site_input",
        help="Directory containing segment_plan_event/noise/unlabel CSVs.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help="Explicit segment plan CSV. Can be repeated. Overrides --plan_dir defaults.",
    )
    parser.add_argument(
        "--out_dir",
        default="output_npy/final/raw_frequency_analysis",
        help="Output directory for CSV/PNG summaries.",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=[],
        help="Optional site filter. Can be repeated, e.g. --site pohang.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label filter by name or id. Can be repeated, e.g. --label event.",
    )
    parser.add_argument("--fs", type=float, default=None, help="Fallback raw sampling rate.")
    parser.add_argument(
        "--site_fs",
        action="append",
        default=[],
        help="Site-specific raw fs mapping. Example: --site_fs pohang:1000.",
    )
    parser.add_argument("--site_col", default="site")
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--tdms-group", default=None)
    parser.add_argument("--tdms-channel", default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. Default analyzes all rows.",
    )
    return parser.parse_args()


def parse_site_fs(items: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        site, fs = str(item).split(":", 1)
        out[site.strip().lower()] = float(fs)
    return out


def load_plans(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        paths = [Path(p) for p in args.csv]
    else:
        plan_dir = Path(args.plan_dir)
        paths = [plan_dir / name for name in DEFAULT_PLAN_FILES]

    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["__source_csv__"] = str(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if args.site:
        wanted = {s.strip().lower() for s in args.site}
        out = out[out[args.site_col].astype(str).str.strip().str.lower().isin(wanted)].copy()

    if args.label:
        wanted_labels = {normalize_label_token(x) for x in args.label}
        labels = out.apply(lambda r: normalize_label_token(infer_label_from_row(r, args.label_col)), axis=1)
        out = out[labels.isin(wanted_labels)].copy()

    if args.max_rows is not None:
        out = out.head(int(args.max_rows)).copy()

    return out.reset_index(drop=True)


def normalize_label_token(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return LABEL_ID_TO_DIR[int(value)].split("_", 1)[1]
    s = str(value).strip().lower()
    if s in {"0", "noise"}:
        return "noise"
    if s in {"1", "event"}:
        return "event"
    if s in {"2", "-1", "unlabel", "unlabeled", "unlabelled"}:
        return "unlabel"
    return s


def plot_frequency_by_site_label(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    label_order = ["event", "noise", "unlabel"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), dpi=150, sharex=False)
    for ax, label_name in zip(axes, label_order):
        sub = df[df["site_label"].astype(str).str.endswith("|" + label_name)]
        for site_label, g in sub.groupby("site_label"):
            site = str(site_label).split("|", 1)[0]
            # A site/label may have multiple raw fs or window lengths. Plot each curve.
            for (saved_fs, fft_n), gg in g.groupby(["saved_fs", "fft_n"]):
                gg = gg.sort_values("freq_hz")
                ax.plot(
                    gg["freq_hz"],
                    gg["power_fraction"],
                    linewidth=1.2,
                    label=f"{site} fs={saved_fs:g} n={int(fft_n)}",
                )
        ax.set_title(label_name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power fraction")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.suptitle("Raw Frequency Energy by Site and Label")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_plans(args)
    site_fs_map = parse_site_fs(args.site_fs)

    freq_by_site_label: Dict[str, Dict[str, Any]] = {}
    freq_by_site: Dict[str, Dict[str, Any]] = {}
    errors = []

    t0 = time.time()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Raw frequency"):
        try:
            site = sanitize_site_name(str(row[args.site_col]))
            label_id = infer_label_from_row(row, args.label_col)
            label_name = LABEL_ID_TO_DIR[int(label_id)].split("_", 1)[1]
            raw_path = resolve_file_path(row)
            start_sec, end_sec = resolve_start_end_sec(row)
            ch_start, ch_end = resolve_channel_range(row)
            original_fs = resolve_original_fs(row, site, site_fs_map, args.fs)

            arr = read_raw_window(
                file_path=raw_path,
                start_sec=start_sec,
                end_sec=end_sec,
                ch_start=ch_start,
                ch_end=ch_end,
                original_fs=original_fs,
                tdms_group=args.tdms_group,
                tdms_channel=args.tdms_channel,
            )
            arr = np.asarray(arr, dtype=np.float32)

            update_frequency_stats(freq_by_site, site, arr, original_fs)
            update_frequency_stats(freq_by_site_label, f"{site}|{label_name}", arr, original_fs)

            if (idx + 1) % int(args.log_every) == 0:
                elapsed = time.time() - t0
                print(
                    f"processed={idx + 1}/{len(df)} failed={len(errors)} "
                    f"elapsed_min={elapsed / 60.0:.2f}",
                    flush=True,
                )
        except Exception as exc:
            errors.append(
                {
                    "row_index": int(idx),
                    "site": row.get(args.site_col, ""),
                    "label": row.get("label", ""),
                    "file_path": row.get("file_path", ""),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "source_csv": row.get("__source_csv__", ""),
                }
            )

    site_df = frequency_stats_to_frame(freq_by_site, group_col="site")
    site_label_df = frequency_stats_to_frame(freq_by_site_label, group_col="site_label")

    site_csv = out_dir / "raw_frequency_energy_by_site.csv"
    site_label_csv = out_dir / "raw_frequency_energy_by_site_label.csv"
    site_df.to_csv(site_csv, index=False)
    site_label_df.to_csv(site_label_csv, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "raw_frequency_errors.csv", index=False)

    png_path = out_dir / "raw_frequency_energy_by_site_label.png"
    if len(site_label_df):
        plot_frequency_by_site_label(site_label_csv, png_path)

    summary = {
        "input_rows": int(len(df)),
        "failed_rows": int(len(errors)),
        "success_rows": int(len(df) - len(errors)),
        "elapsed_sec": float(time.time() - t0),
        "site_csv": str(site_csv),
        "site_label_csv": str(site_label_csv),
        "site_label_png": str(png_path),
        "sites": sorted(site_df["site"].astype(str).unique().tolist()) if len(site_df) else [],
        "site_labels": sorted(site_label_df["site_label"].astype(str).unique().tolist())
        if len(site_label_df)
        else [],
    }
    (out_dir / "raw_frequency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
