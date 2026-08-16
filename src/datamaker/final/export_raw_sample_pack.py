#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from datamaker.final.build_npy_dataset import (
    LABEL_ID_TO_DIR,
    infer_label_from_row,
    maybe_resample,
    read_raw_window,
    resolve_channel_range,
    resolve_file_path,
    resolve_original_fs,
    resolve_start_end_sec,
    sanitize_site_name,
)


DEFAULT_PLAN_FILES = [
    "segment_plan_event.csv",
    "segment_plan_noise.csv",
    "segment_plan_unlabel.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a small raw .npy sample pack for local visualization. "
            "No detrend, no frequency filter, no RMS normalization, no AGC."
        )
    )
    parser.add_argument("--plan_dir", default="outputs/final_3site_input")
    parser.add_argument("--out_dir", default="output_npy/visualize_raw_samples")
    parser.add_argument("--samples_per_site_label", type=int, default=20)
    parser.add_argument("--target_fs", type=float, default=None,
                        help="Optional resampling rate. Omit to preserve original fs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site", action="append", default=[])
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--site_col", default="site")
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--site_fs", action="append", default=[])
    parser.add_argument("--tdms-group", default=None)
    parser.add_argument("--tdms-channel", default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def parse_site_fs(items: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        site, fs = str(item).split(":", 1)
        out[site.strip().lower()] = float(fs)
    return out


def normalize_label_name(value: Any) -> str:
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


def load_plans(args: argparse.Namespace) -> pd.DataFrame:
    plan_dir = Path(args.plan_dir)
    frames = []
    for name in DEFAULT_PLAN_FILES:
        path = plan_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["__source_csv__"] = str(path)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    label_ids = df_all.apply(lambda r: infer_label_from_row(r, args.label_col), axis=1)
    df_all["__label_id__"] = label_ids.astype(int)
    df_all["__label_name__"] = df_all["__label_id__"].map(
        lambda x: LABEL_ID_TO_DIR[int(x)].split("_", 1)[1]
    )
    df_all["__site__"] = df_all[args.site_col].astype(str).str.strip().str.lower()

    if args.site:
        wanted_sites = {str(s).strip().lower() for s in args.site}
        df_all = df_all[df_all["__site__"].isin(wanted_sites)].copy()

    if args.label:
        wanted_labels = {normalize_label_name(x) for x in args.label}
        df_all = df_all[df_all["__label_name__"].isin(wanted_labels)].copy()

    return df_all.reset_index(drop=True)


def choose_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    selected = []
    for (site, label_name), group in df.groupby(["__site__", "__label_name__"], sort=True):
        take = min(int(n), len(group))
        selected.append(group.sample(n=take, random_state=seed).sort_index())
    if not selected:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(selected, ignore_index=True)


def make_output_path(out_dir: Path, site: str, label_id: int, label_name: str, idx: int) -> Path:
    label_dir = f"{label_id}_{label_name}"
    return out_dir / site / label_dir / f"{idx:04d}.npy"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    site_fs_map = parse_site_fs(args.site_fs)

    df_all = load_plans(args)
    selected = choose_rows(df_all, args.samples_per_site_label, args.seed)

    manifest_rows = []
    error_rows = []
    counters: Dict[str, int] = {}

    for selected_index, row in tqdm(selected.iterrows(), total=len(selected), desc="Export raw samples"):
        try:
            site = sanitize_site_name(str(row["__site__"]))
            label_id = int(row["__label_id__"])
            label_name = str(row["__label_name__"])
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
            arr = maybe_resample(arr, original_fs=original_fs, target_fs=args.target_fs)
            saved_fs = float(args.target_fs) if args.target_fs is not None else float(original_fs)
            arr = np.asarray(arr, dtype=np.float32 if args.dtype == "float32" else np.float64)

            key = f"{site}|{label_name}"
            sample_idx = counters.get(key, 0)
            counters[key] = sample_idx + 1
            out_path = make_output_path(out_dir, site, label_id, label_name, sample_idx)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, arr)

            manifest_rows.append({
                "pack_index": int(len(manifest_rows)),
                "selected_source_index": int(selected_index),
                "site": site,
                "label": label_id,
                "label_name": label_name,
                "npy_path": str(out_path),
                "raw_file_path": str(raw_path),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "ch_start": int(ch_start),
                "ch_end": int(ch_end),
                "num_channels": int(arr.shape[0]),
                "original_fs": float(original_fs),
                "saved_fs": float(saved_fs),
                "target_fs": None if args.target_fs is None else float(args.target_fs),
                "num_samples": int(arr.shape[-1]),
                "shape0": int(arr.shape[0]),
                "shape1": int(arr.shape[1]),
                "preprocess": "raw_window_only",
                "detrend": False,
                "filter": False,
                "rms_normalize": False,
                "agc": False,
                "source_csv": row.get("__source_csv__", ""),
            })
        except Exception as exc:
            error_rows.append({
                "selected_source_index": int(selected_index),
                "site": row.get("__site__", ""),
                "label_name": row.get("__label_name__", ""),
                "file_path": row.get("file_path", ""),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "raw_sample_pack_manifest.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(out_dir / "raw_sample_pack_errors.csv", index=False)

    counts_by_site_label = {}
    if len(manifest):
        counts = manifest.groupby(["site", "label_name"]).size()
        counts_by_site_label = {f"{site}|{label}": int(count) for (site, label), count in counts.items()}

    summary = {
        "out_dir": str(out_dir),
        "plan_dir": str(args.plan_dir),
        "samples_per_site_label_requested": int(args.samples_per_site_label),
        "target_fs": None if args.target_fs is None else float(args.target_fs),
        "n_selected": int(len(selected)),
        "n_success": int(len(manifest_rows)),
        "n_failed": int(len(error_rows)),
        "counts_by_site_label": counts_by_site_label,
        "preprocess": {
            "raw_window_only": True,
            "detrend": False,
            "filter": False,
            "rms_normalize": False,
            "agc": False,
            "resample_only_if_target_fs_given": args.target_fs is not None,
        },
    }
    (out_dir / "raw_sample_pack_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
