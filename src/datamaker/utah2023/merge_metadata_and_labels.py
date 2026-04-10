#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge metadata_raw.csv and patch_labels_v2.csv into all_samples.csv.

Recommended placement:
  - metadata_raw.csv:
      <dataset_root>/metadata/metadata_raw.csv
  - patch_labels_v2.csv:
      <dataset_root>/metadata/patch_labels_v2.csv
  - output:
      <dataset_root>/metadata/all_samples.csv

Default label mapping:
  noise  -> 0
  event  -> 1
  ignore -> 2
"""

import argparse
from pathlib import Path
import pandas as pd


LABEL_MAP = {
    "noise": 0,
    "event": 1,
    "ignore": 2,
    "ignored": 2,
    "unlabel": 2,
    "unlabeled": 2,
}


LABEL_NAME_MAP = {
    0: "noise",
    1: "event",
    2: "unlabel",
}


def parse_args():
    p = argparse.ArgumentParser(description="Merge metadata_raw.csv and patch_labels_v2.csv into all_samples.csv")
    p.add_argument("--metadata_raw_csv", type=str, default=None,
                   help="Path to metadata_raw.csv. Default: <dataset_root>/metadata/metadata_raw.csv")
    p.add_argument("--patch_labels_csv", type=str, default=None,
                   help="Path to patch_labels_v2.csv. Default: <dataset_root>/metadata/patch_labels_v2.csv")
    p.add_argument("--out_csv", type=str, default=None,
                   help="Path to all_samples.csv. Default: <dataset_root>/metadata/all_samples.csv")
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Dataset root. If set, paths default under <dataset_root>/metadata/")
    p.add_argument("--site", type=str, default=None,
                   help="Optional override site/domain")
    p.add_argument("--drop_label_reason", action="store_true")
    return p.parse_args()


def resolve_paths(args):
    if args.dataset_root is not None:
        meta_dir = Path(args.dataset_root) / "metadata"
        metadata_raw_csv = Path(args.metadata_raw_csv) if args.metadata_raw_csv else meta_dir / "metadata_raw.csv"
        patch_labels_csv = Path(args.patch_labels_csv) if args.patch_labels_csv else meta_dir / "patch_labels_v2.csv"
        out_csv = Path(args.out_csv) if args.out_csv else meta_dir / "all_samples.csv"
    else:
        if args.metadata_raw_csv is None or args.patch_labels_csv is None or args.out_csv is None:
            raise ValueError("Either --dataset_root or all of --metadata_raw_csv/--patch_labels_csv/--out_csv must be provided.")
        metadata_raw_csv = Path(args.metadata_raw_csv)
        patch_labels_csv = Path(args.patch_labels_csv)
        out_csv = Path(args.out_csv)
    return metadata_raw_csv, patch_labels_csv, out_csv


def main():
    args = parse_args()
    metadata_raw_csv, patch_labels_csv, out_csv = resolve_paths(args)

    if not metadata_raw_csv.exists():
        raise FileNotFoundError(f"metadata_raw.csv not found: {metadata_raw_csv}")
    if not patch_labels_csv.exists():
        raise FileNotFoundError(f"patch_labels_v2.csv not found: {patch_labels_csv}")

    meta_df = pd.read_csv(metadata_raw_csv)
    lbl_df = pd.read_csv(patch_labels_csv)

    meta_required = ["shot_id", "segment_index", "split_id", "npy_path"]
    lbl_required = ["shot_id", "segment_index", "split_id", "label"]
    for c in meta_required:
        if c not in meta_df.columns:
            raise ValueError(f"metadata_raw.csv missing column: {c}")
    for c in lbl_required:
        if c not in lbl_df.columns:
            raise ValueError(f"patch_labels_v2.csv missing column: {c}")

    # normalize types
    meta_df = meta_df.copy()
    lbl_df = lbl_df.copy()

    meta_df["shot_id"] = meta_df["shot_id"].astype(str)
    lbl_df["shot_id"] = lbl_df["shot_id"].astype(str)

    # make shot_id consistent: add .sgy if label csv is missing it
    lbl_df["shot_id"] = lbl_df["shot_id"].apply(lambda x: x if x.endswith(".sgy") else x + ".sgy")

    meta_df["segment_index"] = meta_df["segment_index"].astype(int)
    lbl_df["segment_index"] = lbl_df["segment_index"].astype(int)
    meta_df["split_id"] = meta_df["split_id"].astype(int)
    lbl_df["split_id"] = lbl_df["split_id"].astype(int)

    # reduce label df to needed columns + optional label_reason
    keep_cols = ["shot_id", "segment_index", "split_id", "label"]
    if "label_reason" in lbl_df.columns and not args.drop_label_reason:
        keep_cols.append("label_reason")
    lbl_df = lbl_df[keep_cols].copy()

    # sanity check duplicate keys
    dup_meta = meta_df.duplicated(subset=["shot_id", "segment_index", "split_id"]).sum()
    dup_lbl = lbl_df.duplicated(subset=["shot_id", "segment_index", "split_id"]).sum()
    if dup_meta > 0:
        raise ValueError(f"metadata_raw.csv has duplicated keys: {dup_meta}")
    if dup_lbl > 0:
        raise ValueError(f"patch_labels_v2.csv has duplicated keys: {dup_lbl}")

    merged = meta_df.merge(
        lbl_df,
        on=["shot_id", "segment_index", "split_id"],
        how="left",
        validate="one_to_one",
    )

    # unlabeled if not found
    merged["label"] = merged["label"].fillna("ignore").astype(str).str.strip().str.lower()
    merged["label_id"] = merged["label"].map(LABEL_MAP)

    bad = merged["label_id"].isna()
    if bad.any():
        bad_vals = sorted(merged.loc[bad, "label"].unique().tolist())
        raise ValueError(f"Unknown labels after merge: {bad_vals}")

    merged["label"] = merged["label_id"].astype(int)
    merged["label_name"] = merged["label"].map(LABEL_NAME_MAP)

    # optional site/domain override
    if args.site is not None:
        merged["site"] = args.site
        merged["domain"] = args.site

    # final column order
    preferred_cols = [
        "global_index",
        "sample_index_within_class",
        "site",
        "domain",
        "label",
        "label_name",
        "npy_path",
        "raw_file_path",
        "file_name",
        "file_stem",
        "group_id",
        "shot_id",
        "segment_index",
        "split_id",
        "split_name",
        "view",
        "start_sec",
        "end_sec",
        "duration_sec",
        "ch_start",
        "ch_end",
        "num_channels",
        "original_fs",
        "saved_fs",
        "num_samples",
        "shape0",
        "shape1",
        "label_reason",
    ]
    final_cols = [c for c in preferred_cols if c in merged.columns] + [c for c in merged.columns if c not in preferred_cols]
    merged = merged[final_cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {out_csv}")
    print("[INFO] label counts:")
    print(merged["label_name"].value_counts(dropna=False))
    print("[INFO] rows:", len(merged))


if __name__ == "__main__":
    main()
