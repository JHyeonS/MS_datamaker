#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create experiment-ready CSV splits for DAS microseismic research.

Experiments
-----------
1) stage1_pohang_only
2) stage1_utah_only
3) stage2_joint
4) stage3_pohang_to_utah
5) stage3_utah_to_pohang

Expected input columns
----------------------
- site
- label   (0=noise, 1=event, 2=unlabeled)
- group_id or file_stem

Outputs per experiment directory
--------------------------------
- pretrain.csv
- train.csv
- val.csv
- test.csv
- summary.json

Main idea
---------
1) group-aware split
2) satisfy minimum event/noise quotas FIRST
3) then fill remaining groups to roughly match ratios
4) search seeds until a valid split is found
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NOISE_LABEL = 0
EVENT_LABEL = 1
UNLABELED_LABEL = 2

VALID_LABELED = {0, 1}
VALID_ALL = {0, 1, 2}


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
    split_key = pd.Series(split_key, index=df.index).astype(str)

    if np.any(split_key == ""):
        raise ValueError("Failed to build split_key. Need group_id or file_stem.")

    df["split_key"] = df["site"].astype(str) + "::" + split_key
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


def labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"].isin(VALID_LABELED)].copy()


def subset_by_keys(df: pd.DataFrame, keys: set) -> pd.DataFrame:
    if len(keys) == 0:
        return df.iloc[0:0].copy()
    return df[df["split_key"].isin(keys)].copy()


def subset_excluding_keys(df: pd.DataFrame, keys: set) -> pd.DataFrame:
    if len(keys) == 0:
        return df.copy()
    return df[~df["split_key"].isin(keys)].copy()


def summarize_df(df: pd.DataFrame) -> Dict:
    out: Dict = {
        "n_rows": int(len(df)),
        "n_groups": int(df["split_key"].nunique()) if "split_key" in df.columns else 0,
    }

    if "site" in df.columns:
        out["site_counts"] = df["site"].value_counts(dropna=False).to_dict()

    if "label" in df.columns:
        out["label_counts"] = {
            str(k): int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()
        }

    if "label_name" in df.columns:
        out["label_name_counts"] = df["label_name"].value_counts(dropna=False).to_dict()

    return out


def group_label_table(df_labeled: pd.DataFrame) -> pd.DataFrame:
    g = (
        df_labeled.groupby(["split_key", "label"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    if NOISE_LABEL not in g.columns:
        g[NOISE_LABEL] = 0
    if EVENT_LABEL not in g.columns:
        g[EVENT_LABEL] = 0

    g["n_total"] = g[NOISE_LABEL] + g[EVENT_LABEL]
    g["has_noise"] = (g[NOISE_LABEL] > 0).astype(int)
    g["has_event"] = (g[EVENT_LABEL] > 0).astype(int)

    return g[["split_key", NOISE_LABEL, EVENT_LABEL, "n_total", "has_noise", "has_event"]].copy()


def get_split_stats(df_labeled: pd.DataFrame, train_keys: set, val_keys: set, test_keys: set) -> Dict:
    train_df = subset_by_keys(df_labeled, train_keys)
    val_df = subset_by_keys(df_labeled, val_keys)
    test_df = subset_by_keys(df_labeled, test_keys)

    def counts(df: pd.DataFrame) -> Dict[str, int]:
        return {
            "noise": int((df["label"] == NOISE_LABEL).sum()),
            "event": int((df["label"] == EVENT_LABEL).sum()),
            "rows": int(len(df)),
            "groups": int(df["split_key"].nunique()),
        }

    return {
        "train": counts(train_df),
        "val": counts(val_df),
        "test": counts(test_df),
    }


def check_min_requirements(stats: Dict, min_reqs: Dict) -> bool:
    for split_name in ["train", "val", "test"]:
        req = min_reqs.get(split_name, {})
        got = stats.get(split_name, {})

        if got.get("noise", 0) < req.get("noise", 0):
            return False
        if got.get("event", 0) < req.get("event", 0):
            return False
        if got.get("rows", 0) < req.get("rows", 0):
            return False
        if got.get("groups", 0) < req.get("groups", 0):
            return False

    return True


def _required_splits(train_ratio: float, val_ratio: float, test_ratio: float) -> List[str]:
    out = []
    if train_ratio > 0:
        out.append("train")
    if val_ratio > 0:
        out.append("val")
    if test_ratio > 0:
        out.append("test")
    return out


def _build_target(group_df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict:
    total_noise = int(group_df[NOISE_LABEL].sum())
    total_event = int(group_df[EVENT_LABEL].sum())
    total_all = int(group_df["n_total"].sum())

    return {
        "train": {
            "noise": total_noise * train_ratio,
            "event": total_event * train_ratio,
            "rows": total_all * train_ratio,
        },
        "val": {
            "noise": total_noise * val_ratio,
            "event": total_event * val_ratio,
            "rows": total_all * val_ratio,
        },
        "test": {
            "noise": total_noise * test_ratio,
            "event": total_event * test_ratio,
            "rows": total_all * test_ratio,
        },
    }


def _init_assigned() -> Dict:
    return {
        "train": {"noise": 0, "event": 0, "rows": 0, "keys": []},
        "val": {"noise": 0, "event": 0, "rows": 0, "keys": []},
        "test": {"noise": 0, "event": 0, "rows": 0, "keys": []},
    }


def _add_group(assigned: Dict, split_name: str, row: pd.Series) -> None:
    assigned[split_name]["noise"] += int(row[NOISE_LABEL])
    assigned[split_name]["event"] += int(row[EVENT_LABEL])
    assigned[split_name]["rows"] += int(row["n_total"])
    assigned[split_name]["keys"].append(row["split_key"])


def _assignment_score(
    split_name: str,
    assigned: Dict,
    target: Dict,
    row: pd.Series,
) -> float:
    new_noise = assigned[split_name]["noise"] + int(row[NOISE_LABEL])
    new_event = assigned[split_name]["event"] + int(row[EVENT_LABEL])
    new_rows = assigned[split_name]["rows"] + int(row["n_total"])

    score = 0.0
    score += abs(new_noise - target[split_name]["noise"]) / max(1.0, target[split_name]["noise"] + 1e-6)
    score += abs(new_event - target[split_name]["event"]) / max(1.0, target[split_name]["event"] + 1e-6)
    score += 0.5 * abs(new_rows - target[split_name]["rows"]) / max(1.0, target[split_name]["rows"] + 1e-6)
    return float(score)


def _pick_group_for_deficit(
    candidates: List[pd.Series],
    deficit_key: str,
    deficit_value: int,
    prefer_small: bool,
) -> pd.Series:
    """
    Pick a group to satisfy deficit:
    - if possible, choose one with count close to deficit
    - prefer small groups for val/test, large groups for train
    """
    if len(candidates) == 0:
        raise RuntimeError("No candidate groups available.")

    if deficit_key == "event":
        count_fn = lambda r: int(r[EVENT_LABEL])
    elif deficit_key == "noise":
        count_fn = lambda r: int(r[NOISE_LABEL])
    else:
        raise ValueError(f"Unknown deficit_key: {deficit_key}")

    # candidates with enough count
    enough = [r for r in candidates if count_fn(r) >= deficit_value and count_fn(r) > 0]
    positive = [r for r in candidates if count_fn(r) > 0]

    if len(positive) == 0:
        raise RuntimeError(f"No positive candidates for deficit_key={deficit_key}")

    if len(enough) > 0:
        if prefer_small:
            return min(enough, key=lambda r: (count_fn(r), int(r["n_total"])))
        return max(enough, key=lambda r: (count_fn(r), int(r["n_total"])))

    # otherwise choose best available
    if prefer_small:
        return min(positive, key=lambda r: (abs(count_fn(r) - deficit_value), int(r["n_total"])))
    return max(positive, key=lambda r: (count_fn(r), int(r["n_total"])))


def quota_first_group_split(
    df_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_reqs: Dict,
) -> Tuple[set, set, set]:
    """
    3-way quota-first split:
    1) satisfy event quotas
    2) satisfy noise quotas
    3) fill remaining groups by ratio score
    """
    if len(df_labeled) == 0:
        raise ValueError("No labeled rows to split.")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    group_df = group_label_table(df_labeled)
    target = _build_target(group_df, train_ratio, val_ratio, test_ratio)
    assigned = _init_assigned()
    required = _required_splits(train_ratio, val_ratio, test_ratio)

    rng = np.random.default_rng(seed)
    group_df = group_df.iloc[rng.permutation(len(group_df))].reset_index(drop=True)

    used_keys = set()

    def get_available(mask_fn) -> List[pd.Series]:
        rows = []
        for _, row in group_df.iterrows():
            if row["split_key"] in used_keys:
                continue
            if mask_fn(row):
                rows.append(row)
        return rows

    # ---------- Step 1: satisfy event quotas ----------
    # smaller splits first to avoid train swallowing everything
    event_order = [s for s in ["val", "test", "train"] if s in required]

    for split_name in event_order:
        needed = int(min_reqs.get(split_name, {}).get("event", 0))
        while assigned[split_name]["event"] < needed:
            deficit = needed - assigned[split_name]["event"]
            candidates = get_available(lambda r: int(r[EVENT_LABEL]) > 0)

            if len(candidates) == 0:
                raise RuntimeError(f"Cannot satisfy event quota for split={split_name}")

            row = _pick_group_for_deficit(
                candidates=candidates,
                deficit_key="event",
                deficit_value=deficit,
                prefer_small=(split_name != "train"),
            )
            _add_group(assigned, split_name, row)
            used_keys.add(row["split_key"])

    # ---------- Step 2: satisfy noise quotas ----------
    noise_order = [s for s in ["val", "test", "train"] if s in required]

    for split_name in noise_order:
        needed = int(min_reqs.get(split_name, {}).get("noise", 0))
        while assigned[split_name]["noise"] < needed:
            deficit = needed - assigned[split_name]["noise"]
            candidates = get_available(lambda r: int(r[NOISE_LABEL]) > 0)

            if len(candidates) == 0:
                raise RuntimeError(f"Cannot satisfy noise quota for split={split_name}")

            row = _pick_group_for_deficit(
                candidates=candidates,
                deficit_key="noise",
                deficit_value=deficit,
                prefer_small=(split_name != "train"),
            )
            _add_group(assigned, split_name, row)
            used_keys.add(row["split_key"])

    # ---------- Step 3: fill remaining groups by ratio ----------
    remaining = group_df[~group_df["split_key"].isin(list(used_keys))].copy()
    if len(remaining) > 0:
        remaining = remaining.sort_values("n_total", ascending=False).reset_index(drop=True)

    for _, row in remaining.iterrows():
        best_split = None
        best_score = None
        for split_name in required:
            score = _assignment_score(split_name, assigned, target, row)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        _add_group(assigned, best_split, row)
        used_keys.add(row["split_key"])

    train_keys = set(assigned["train"]["keys"])
    val_keys = set(assigned["val"]["keys"])
    test_keys = set(assigned["test"]["keys"])

    return train_keys, val_keys, test_keys


def quota_first_group_split_train_val(
    df_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_reqs: Dict,
) -> Tuple[set, set]:
    """
    2-way quota-first split for source train/val in stage3.
    """
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError("train_ratio + val_ratio must be > 0")

    tr = train_ratio / total
    vr = val_ratio / total

    min_reqs_2 = {
        "train": min_reqs.get("train", {}),
        "val": min_reqs.get("val", {}),
        "test": {},
    }

    train_keys, val_keys, _ = quota_first_group_split(
        df_labeled=df_labeled,
        train_ratio=tr,
        val_ratio=vr,
        test_ratio=0.0,
        seed=seed,
        min_reqs=min_reqs_2,
    )
    return train_keys, val_keys


def search_valid_split_with_constraints(
    df_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    base_seed: int,
    max_tries: int,
    min_reqs: Dict,
) -> Tuple[set, set, set, int, Dict]:
    last_stats = None
    last_error = None

    for offset in range(max_tries):
        seed = base_seed + offset
        try:
            train_keys, val_keys, test_keys = quota_first_group_split(
                df_labeled=df_labeled,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                min_reqs=min_reqs,
            )
            stats = get_split_stats(df_labeled, train_keys, val_keys, test_keys)
            last_stats = stats

            if check_min_requirements(stats, min_reqs):
                return train_keys, val_keys, test_keys, seed, stats

        except Exception as e:
            last_error = str(e)
            continue

    msg = {
        "message": "Could not find a valid split satisfying constraints.",
        "base_seed": base_seed,
        "max_tries": max_tries,
        "min_requirements": min_reqs,
        "last_stats": last_stats,
        "last_error": last_error,
    }
    raise RuntimeError(json.dumps(msg, ensure_ascii=False, indent=2))


def search_valid_train_val_split_with_constraints(
    df_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    base_seed: int,
    max_tries: int,
    min_reqs: Dict,
) -> Tuple[set, set, int, Dict]:
    last_stats = None
    last_error = None

    for offset in range(max_tries):
        seed = base_seed + offset
        try:
            train_keys, val_keys = quota_first_group_split_train_val(
                df_labeled=df_labeled,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
                min_reqs=min_reqs,
            )
            stats = get_split_stats(df_labeled, train_keys, val_keys, set())
            last_stats = stats

            reduced_reqs = {
                "train": min_reqs.get("train", {}),
                "val": min_reqs.get("val", {}),
                "test": {},
            }

            if check_min_requirements(stats, reduced_reqs):
                return train_keys, val_keys, seed, stats

        except Exception as e:
            last_error = str(e)
            continue

    msg = {
        "message": "Could not find a valid train/val split satisfying constraints.",
        "base_seed": base_seed,
        "max_tries": max_tries,
        "min_requirements": min_reqs,
        "last_stats": last_stats,
        "last_error": last_error,
    }
    raise RuntimeError(json.dumps(msg, ensure_ascii=False, indent=2))


def build_min_requirements(
    n_noise: int,
    n_event: int,
    mode: str,
    min_event_train: int,
    min_event_val: int,
    min_event_test: int,
    min_noise_train: int,
    min_noise_val: int,
    min_noise_test: int,
) -> Dict:
    req = {
        "train": {"noise": 0, "event": 0, "rows": 0, "groups": 1},
        "val": {"noise": 0, "event": 0, "rows": 0, "groups": 1},
        "test": {"noise": 0, "event": 0, "rows": 0, "groups": 1},
    }

    req["train"]["event"] = min(min_event_train, n_event)
    req["val"]["event"] = min(min_event_val, n_event)
    req["test"]["event"] = min(min_event_test, n_event)

    req["train"]["noise"] = min(min_noise_train, n_noise)
    req["val"]["noise"] = min(min_noise_val, n_noise)
    req["test"]["noise"] = min(min_noise_test, n_noise)

    if mode == "split2":
        req["test"] = {}

    return req


def default_constraints_for_dataset(
    df_labeled: pd.DataFrame,
    mode: str,
) -> Dict:
    n_noise = int((df_labeled["label"] == NOISE_LABEL).sum())
    n_event = int((df_labeled["label"] == EVENT_LABEL).sum())

    if mode == "split3":
        return build_min_requirements(
            n_noise=n_noise,
            n_event=n_event,
            mode=mode,
            min_event_train=max(5, min(20, n_event)),
            min_event_val=max(1, min(10, max(1, n_event // 10))),
            min_event_test=max(1, min(10, max(1, n_event // 10))),
            min_noise_train=max(20, min(100, n_noise)),
            min_noise_val=max(5, min(30, max(1, n_noise // 10))),
            min_noise_test=max(5, min(30, max(1, n_noise // 10))),
        )

    return build_min_requirements(
        n_noise=n_noise,
        n_event=n_event,
        mode=mode,
        min_event_train=max(5, min(20, n_event)),
        min_event_val=max(1, min(10, max(1, n_event // 10))),
        min_event_test=0,
        min_noise_train=max(20, min(100, n_noise)),
        min_noise_val=max(5, min(30, max(1, n_noise // 10))),
        min_noise_test=0,
    )


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
    max_tries: int,
    min_reqs_override: Optional[Dict] = None,
) -> None:
    exp_name = f"stage1_{site_name.lower()}_only"
    out_dir = out_root / exp_name

    df_site = df_all[df_all["site"].astype(str).str.lower() == site_name.lower()].copy()
    df_site_labeled = labeled_only(df_site)

    if len(df_site_labeled) == 0:
        raise ValueError(f"No labeled rows found for site={site_name}")

    min_reqs = min_reqs_override or default_constraints_for_dataset(df_site_labeled, mode="split3")

    train_keys, val_keys, test_keys, used_seed, stats = search_valid_split_with_constraints(
        df_labeled=df_site_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        base_seed=seed,
        max_tries=max_tries,
        min_reqs=min_reqs,
    )

    train_df = subset_by_keys(df_site_labeled, train_keys)
    val_df = subset_by_keys(df_site_labeled, val_keys)
    test_df = subset_by_keys(df_site_labeled, test_keys)
    pretrain_df = subset_excluding_keys(df_site, test_keys)

    summary = {
        "experiment": exp_name,
        "seed": int(used_seed),
        "site": site_name,
        "description": "Site-specific experiment. Pretrain on same-site non-test groups. Fine-tune/evaluate on same-site labeled splits. Quota-first constrained group-aware split.",
        "min_requirements": min_reqs,
        "split_stats": stats,
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
    max_tries: int,
    min_reqs_override: Optional[Dict] = None,
) -> None:
    exp_name = "stage2_joint"
    out_dir = out_root / exp_name

    df_labeled = labeled_only(df_all)
    min_reqs = min_reqs_override or default_constraints_for_dataset(df_labeled, mode="split3")

    train_keys, val_keys, test_keys, used_seed, stats = search_valid_split_with_constraints(
        df_labeled=df_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        base_seed=seed,
        max_tries=max_tries,
        min_reqs=min_reqs,
    )

    train_df = subset_by_keys(df_labeled, train_keys)
    val_df = subset_by_keys(df_labeled, val_keys)
    test_df = subset_by_keys(df_labeled, test_keys)
    pretrain_df = subset_excluding_keys(df_all, test_keys)

    summary = {
        "experiment": exp_name,
        "seed": int(used_seed),
        "description": "Joint Pohang+UTAH experiment. Pretrain on all non-test groups from both sites. Quota-first constrained group-aware split.",
        "min_requirements": min_reqs,
        "split_stats": stats,
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
    max_tries: int,
    source_min_reqs_override: Optional[Dict] = None,
    target_test_min_reqs_override: Optional[Dict] = None,
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

    source_min_reqs = source_min_reqs_override or default_constraints_for_dataset(df_src_labeled, mode="split2")

    train_keys, val_keys, used_seed, stats = search_valid_train_val_split_with_constraints(
        df_labeled=df_src_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        base_seed=seed,
        max_tries=max_tries,
        min_reqs=source_min_reqs,
    )

    train_df = subset_by_keys(df_src_labeled, train_keys)
    val_df = subset_by_keys(df_src_labeled, val_keys)
    test_df = df_tgt_labeled.copy()
    pretrain_df = df_src.copy()

    if target_test_min_reqs_override is None:
        n_noise = int((df_tgt_labeled["label"] == NOISE_LABEL).sum())
        n_event = int((df_tgt_labeled["label"] == EVENT_LABEL).sum())
        target_test_min_reqs = {
            "train": {},
            "val": {},
            "test": {
                "noise": min(30, n_noise),
                "event": min(10, n_event),
                "rows": 1,
                "groups": 1,
            },
        }
    else:
        target_test_min_reqs = target_test_min_reqs_override

    target_stats = {
        "train": {},
        "val": {},
        "test": {
            "noise": int((test_df["label"] == NOISE_LABEL).sum()),
            "event": int((test_df["label"] == EVENT_LABEL).sum()),
            "rows": int(len(test_df)),
            "groups": int(test_df["split_key"].nunique()),
        },
    }

    if not check_min_requirements(target_stats, target_test_min_reqs):
        raise RuntimeError(
            json.dumps(
                {
                    "message": f"Target site '{target_site}' test does not satisfy minimum requirements.",
                    "target_test_min_requirements": target_test_min_reqs,
                    "target_test_stats": target_stats,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    summary = {
        "experiment": exp_name,
        "seed": int(used_seed),
        "source_site": source_site,
        "target_site": target_site,
        "description": "Cross-site generalization. Train/val on source labeled only, test on target labeled only, pretrain on source full pool only. Quota-first constrained group-aware split.",
        "source_min_requirements": source_min_reqs,
        "source_split_stats": stats,
        "target_test_min_requirements": target_test_min_reqs,
        "target_test_stats": target_stats,
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
    ap.add_argument("--max_tries", type=int, default=500)

    ap.add_argument("--pohang_name", type=str, default="pohang")
    ap.add_argument("--utah_name", type=str, default="utah")

    ap.add_argument("--min_event_train", type=int, default=-1)
    ap.add_argument("--min_event_val", type=int, default=-1)
    ap.add_argument("--min_event_test", type=int, default=-1)
    ap.add_argument("--min_noise_train", type=int, default=-1)
    ap.add_argument("--min_noise_val", type=int, default=-1)
    ap.add_argument("--min_noise_test", type=int, default=-1)

    args = ap.parse_args()

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    df_all = pd.read_csv(args.all_csv)
    validate_labels(df_all)
    df_all = add_split_key(df_all)

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    def maybe_override(df_labeled: pd.DataFrame, mode: str) -> Optional[Dict]:
        values = [
            args.min_event_train, args.min_event_val, args.min_event_test,
            args.min_noise_train, args.min_noise_val, args.min_noise_test
        ]
        if all(v < 0 for v in values):
            return None

        n_noise = int((df_labeled["label"] == NOISE_LABEL).sum())
        n_event = int((df_labeled["label"] == EVENT_LABEL).sum())

        return build_min_requirements(
            n_noise=n_noise,
            n_event=n_event,
            mode=mode,
            min_event_train=args.min_event_train if args.min_event_train >= 0 else min(20, n_event),
            min_event_val=args.min_event_val if args.min_event_val >= 0 else min(10, n_event),
            min_event_test=args.min_event_test if args.min_event_test >= 0 else min(10, n_event),
            min_noise_train=args.min_noise_train if args.min_noise_train >= 0 else min(100, n_noise),
            min_noise_val=args.min_noise_val if args.min_noise_val >= 0 else min(30, n_noise),
            min_noise_test=args.min_noise_test if args.min_noise_test >= 0 else min(30, n_noise),
        )

    df_pohang = labeled_only(df_all[df_all["site"].astype(str).str.lower() == args.pohang_name.lower()].copy())
    df_utah = labeled_only(df_all[df_all["site"].astype(str).str.lower() == args.utah_name.lower()].copy())
    df_joint = labeled_only(df_all.copy())

    build_stage1_site_only(
        df_all=df_all,
        site_name=args.pohang_name,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_tries=args.max_tries,
        min_reqs_override=maybe_override(df_pohang, "split3"),
    )

    build_stage1_site_only(
        df_all=df_all,
        site_name=args.utah_name,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_tries=args.max_tries,
        min_reqs_override=maybe_override(df_utah, "split3"),
    )

    build_stage2_joint(
        df_all=df_all,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_tries=args.max_tries,
        min_reqs_override=maybe_override(df_joint, "split3"),
    )

    build_stage3_cross_site(
        df_all=df_all,
        source_site=args.pohang_name,
        target_site=args.utah_name,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_tries=args.max_tries,
        source_min_reqs_override=maybe_override(df_pohang, "split2"),
        target_test_min_reqs_override=None,
    )

    build_stage3_cross_site(
        df_all=df_all,
        source_site=args.utah_name,
        target_site=args.pohang_name,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_tries=args.max_tries,
        source_min_reqs_override=maybe_override(df_utah, "split2"),
        target_test_min_reqs_override=None,
    )

    print(f"[DONE] experiment splits saved to: {out_root}")


if __name__ == "__main__":
    main()