#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create experiment-ready CSV splits for DAS microseismic research.

Experiments:
1) stage1_pohang_only
2) stage1_utah_only
3) stage2_joint
4) stage3_pohang_to_utah
5) stage3_utah_to_pohang

Input:
    all_samples.csv

Expected columns:
    - site
    - label   (0=noise, 1=event, -1=unlabeled)
    - group_id or file_stem

Optional columns:
    - label_name
    - npy_path / path / any metadata columns

Outputs for each experiment directory:
    - pretrain.csv
    - train.csv
    - val.csv
    - test.csv
    - summary.json

Design choices:
    - supervised split is group-aware
    - pretrain.csv contains all samples except test groups (default)
    - cross-site: source site -> train/val, target site -> test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


VALID_LABELED = {0, 1}
VALID_ALL = {0, 1, 2}
UNLABELED_VALUE = 2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_split_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "site" not in df.columns:
        raise ValueError("Input CSV must contain column 'site'")

    if "group_id" in df.columns:
        gid = df["group_id"].astype(str)
    else:
        gid = pd.Series(["-1"] * len(df), index=df.index)

    if "file_stem" in df.columns:
        fst = df["file_stem"].astype(str)
    else:
        fst = pd.Series([""] * len(df), index=df.index)

    use_group = gid != "-1"
    split_key = np.where(use_group, gid, fst)

    if np.any(pd.Series(split_key).astype(str) == ""):
        bad_rows = int((pd.Series(split_key).astype(str) == "").sum())
        raise ValueError(
            f"Could not build split_key for {bad_rows} rows. "
            f"Need at least one of ['group_id', 'file_stem']."
        )

    df["split_key"] = df["site"].astype(str) + "::" + pd.Series(split_key, index=df.index).astype(str)
    return df


def validate_labels(df: pd.DataFrame) -> None:
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain column 'label'")

    uniq = set(pd.unique(df["label"]))
    if not uniq.issubset(VALID_ALL):
        raise ValueError(
            f"Unexpected labels found: {sorted(uniq)}. "
            f"Expected subset of {sorted(VALID_ALL)}."
        )


def group_label_table(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per split_key with counts of label 0 / 1.
    """
    g = (
        df_labeled.groupby(["split_key", "label"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    if 0 not in g.columns:
        g[0] = 0
    if 1 not in g.columns:
        g[1] = 0

    g["n_total"] = g[0] + g[1]
    return g[["split_key", 0, 1, "n_total"]].copy()


def greedy_group_split(
    df_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    """
    Group-aware, approximately label-balanced split.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    if len(df_labeled) == 0:
        raise ValueError("No labeled rows to split.")

    group_df = group_label_table(df_labeled)
    rng = np.random.default_rng(seed)

    # shuffle first, then sort by size desc so large groups placed early
    perm = rng.permutation(len(group_df))
    group_df = group_df.iloc[perm].reset_index(drop=True)
    group_df = group_df.sort_values("n_total", ascending=False).reset_index(drop=True)

    total_noise = int(group_df[0].sum())
    total_event = int(group_df[1].sum())
    total_all = int(group_df["n_total"].sum())

    target = {
        "train": {
            0: total_noise * train_ratio,
            1: total_event * train_ratio,
            "n_total": total_all * train_ratio,
        },
        "val": {
            0: total_noise * val_ratio,
            1: total_event * val_ratio,
            "n_total": total_all * val_ratio,
        },
        "test": {
            0: total_noise * test_ratio,
            1: total_event * test_ratio,
            "n_total": total_all * test_ratio,
        },
    }

    assigned = {
        "train": {0: 0, 1: 0, "n_total": 0, "keys": []},
        "val": {0: 0, 1: 0, "n_total": 0, "keys": []},
        "test": {0: 0, 1: 0, "n_total": 0, "keys": []},
    }

    split_names = ["train", "val", "test"]

    for _, row in group_df.iterrows():
        key = row["split_key"]
        g_noise = int(row[0])
        g_event = int(row[1])
        g_total = int(row["n_total"])

        best_split = None
        best_score = None

        for s in split_names:
            # after assigning this group, how far from target?
            new_noise = assigned[s][0] + g_noise
            new_event = assigned[s][1] + g_event
            new_total = assigned[s]["n_total"] + g_total

            # weighted distance to target
            score = (
                abs(new_noise - target[s][0]) / max(1.0, target[s][0] + 1e-6)
                + abs(new_event - target[s][1]) / max(1.0, target[s][1] + 1e-6)
                + 0.5 * abs(new_total - target[s]["n_total"]) / max(1.0, target[s]["n_total"] + 1e-6)
            )

            # mild penalty if split already very full
            fullness = assigned[s]["n_total"] / max(1.0, target[s]["n_total"] + 1e-6)
            score += 0.2 * max(0.0, fullness - 1.0)

            if best_score is None or score < best_score:
                best_score = score
                best_split = s

        assigned[best_split][0] += g_noise
        assigned[best_split][1] += g_event
        assigned[best_split]["n_total"] += g_total
        assigned[best_split]["keys"].append(key)

    train_keys = set(assigned["train"]["keys"])
    val_keys = set(assigned["val"]["keys"])
    test_keys = set(assigned["test"]["keys"])

    # safety: ensure non-empty if possible
    all_keys = set(group_df["split_key"].tolist())
    if len(all_keys) >= 3:
        if len(train_keys) == 0 or len(val_keys) == 0 or len(test_keys) == 0:
            raise RuntimeError("One of train/val/test splits became empty. Try another seed.")

    return train_keys, val_keys, test_keys


def greedy_group_split_train_val(
    df_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[set, set]:
    """
    Group-aware split for source train/val in cross-site experiments.
    """
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError("train_ratio + val_ratio must be > 0")
    tr = train_ratio / total
    vr = val_ratio / total

    train_keys, val_keys, test_keys = greedy_group_split(
        df_labeled=df_labeled,
        train_ratio=tr,
        val_ratio=vr,
        test_ratio=0.0,
        seed=seed,
    )

    # test_keys should normally be empty because target ratio is 0
    if len(test_keys) > 0:
        # move them to train by default
        train_keys = set(train_keys).union(test_keys)

    return train_keys, val_keys


def subset_by_keys(df: pd.DataFrame, keys: set) -> pd.DataFrame:
    if len(keys) == 0:
        return df.iloc[0:0].copy()
    return df[df["split_key"].isin(keys)].copy()


def subset_excluding_keys(df: pd.DataFrame, keys: set) -> pd.DataFrame:
    if len(keys) == 0:
        return df.copy()
    return df[~df["split_key"].isin(keys)].copy()


def labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"].isin(VALID_LABELED)].copy()


def summarize_df(df: pd.DataFrame) -> Dict:
    out: Dict = {
        "n_rows": int(len(df)),
        "n_groups": int(df["split_key"].nunique()) if "split_key" in df.columns else 0,
    }

    if "site" in df.columns:
        out["site_counts"] = df["site"].value_counts(dropna=False).to_dict()

    if "label" in df.columns:
        out["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()}

    if "label_name" in df.columns:
        out["label_name_counts"] = df["label_name"].value_counts(dropna=False).to_dict()

    return out


def save_split_bundle(
    out_dir: Path,
    pretrain_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary: Dict,
) -> None:
    ensure_dir(out_dir)

    pretrain_df.drop(columns=["split_key"], errors="ignore").to_csv(out_dir / "pretrain.csv", index=False)
    train_df.drop(columns=["split_key"], errors="ignore").to_csv(out_dir / "train.csv", index=False)
    val_df.drop(columns=["split_key"], errors="ignore").to_csv(out_dir / "val.csv", index=False)
    test_df.drop(columns=["split_key"], errors="ignore").to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_stage1_site_only(
    df_all: pd.DataFrame,
    site_name: str,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    exp_name = f"stage1_{site_name.lower()}_only"
    out_dir = out_root / exp_name

    df_site = df_all[df_all["site"].astype(str).str.lower() == site_name.lower()].copy()
    df_site_labeled = labeled_only(df_site)

    if len(df_site_labeled) == 0:
        raise ValueError(f"No labeled rows found for site={site_name}")

    train_keys, val_keys, test_keys = greedy_group_split(
        df_labeled=df_site_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_df = subset_by_keys(df_site_labeled, train_keys)
    val_df = subset_by_keys(df_site_labeled, val_keys)
    test_df = subset_by_keys(df_site_labeled, test_keys)

    # pretrain = all samples except supervised test groups
    pretrain_df = subset_excluding_keys(df_site, test_keys)

    summary = {
        "experiment": exp_name,
        "seed": int(seed),
        "site": site_name,
        "description": "Site-specific experiment. Pretrain on same-site non-test groups. Fine-tune/evaluate on same-site labeled splits.",
        "pretrain": summarize_df(pretrain_df),
        "train": summarize_df(train_df),
        "val": summarize_df(val_df),
        "test": summarize_df(test_df),
    }

    save_split_bundle(out_dir, pretrain_df, train_df, val_df, test_df, summary)


def build_stage2_joint(
    df_all: pd.DataFrame,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    exp_name = "stage2_joint"
    out_dir = out_root / exp_name

    df_labeled = labeled_only(df_all)

    train_keys, val_keys, test_keys = greedy_group_split(
        df_labeled=df_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_df = subset_by_keys(df_labeled, train_keys)
    val_df = subset_by_keys(df_labeled, val_keys)
    test_df = subset_by_keys(df_labeled, test_keys)

    # pretrain = all samples except supervised test groups
    pretrain_df = subset_excluding_keys(df_all, test_keys)

    summary = {
        "experiment": exp_name,
        "seed": int(seed),
        "description": "Joint Pohang+UTAH experiment. Pretrain on all non-test groups from both sites.",
        "pretrain": summarize_df(pretrain_df),
        "train": summarize_df(train_df),
        "val": summarize_df(val_df),
        "test": summarize_df(test_df),
    }

    save_split_bundle(out_dir, pretrain_df, train_df, val_df, test_df, summary)


def build_stage3_cross_site(
    df_all: pd.DataFrame,
    source_site: str,
    target_site: str,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    exp_name = f"stage3_{source_site.lower()}_to_{target_site.lower()}"
    out_dir = out_root / exp_name

    df_src = df_all[df_all["site"].astype(str).str.lower() == source_site.lower()].copy()
    df_tgt = df_all[df_all["site"].astype(str).str.lower() == target_site.lower()].copy()

    df_src_labeled = labeled_only(df_src)
    df_tgt_labeled = labeled_only(df_tgt)

    if len(df_src_labeled) == 0:
        raise ValueError(f"No labeled rows found for source_site={source_site}")
    if len(df_tgt_labeled) == 0:
        raise ValueError(f"No labeled rows found for target_site={target_site}")

    train_keys, val_keys = greedy_group_split_train_val(
        df_labeled=df_src_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_df = subset_by_keys(df_src_labeled, train_keys)
    val_df = subset_by_keys(df_src_labeled, val_keys)
    test_df = df_tgt_labeled.copy()

    # pretrain = source site only, excluding nothing beyond source partition test absence
    # strict version: use all source samples except none (target site fully held-out)
    # to avoid source val leakage into representation tuning, one may use source non-test only.
    # here we exclude no target, and keep source-only full pool.
    pretrain_df = df_src.copy()

    summary = {
        "experiment": exp_name,
        "seed": int(seed),
        "source_site": source_site,
        "target_site": target_site,
        "description": "Cross-site generalization. Train/val on source labeled only, test on target labeled only, pretrain on source full pool only.",
        "pretrain": summarize_df(pretrain_df),
        "train": summarize_df(train_df),
        "val": summarize_df(val_df),
        "test": summarize_df(test_df),
    }

    save_split_bundle(out_dir, pretrain_df, train_df, val_df, test_df, summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pohang_name", type=str, default="pohang")
    ap.add_argument("--utah_name", type=str, default="utah")
    args = ap.parse_args()

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    all_csv = Path(args.all_csv)
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    df_all = pd.read_csv(all_csv)
    validate_labels(df_all)
    df_all = add_split_key(df_all)

    # Stage 1
    build_stage1_site_only(
        df_all=df_all,
        site_name=args.pohang_name,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    build_stage1_site_only(
        df_all=df_all,
        site_name=args.utah_name,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Stage 2
    build_stage2_joint(
        df_all=df_all,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Stage 3
    build_stage3_cross_site(
        df_all=df_all,
        source_site=args.pohang_name,
        target_site=args.utah_name,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    build_stage3_cross_site(
        df_all=df_all,
        source_site=args.utah_name,
        target_site=args.pohang_name,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"[DONE] experiment splits saved to: {out_root}")


if __name__ == "__main__":
    main()