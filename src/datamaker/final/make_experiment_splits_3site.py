#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Set, List

import numpy as np
import pandas as pd


NOISE_LABEL = 0
EVENT_LABEL = 1
UNLABEL_LABEL = 2

VALID_LABELED = {NOISE_LABEL, EVENT_LABEL}
VALID_ALL = {NOISE_LABEL, EVENT_LABEL, UNLABEL_LABEL}


# =========================================================
# basic utils
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_df(df: pd.DataFrame) -> None:
    required = ["site", "label", "group_id", "npy_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    uniq = set(pd.unique(df["label"]))
    if not uniq.issubset(VALID_ALL):
        raise ValueError(f"Unexpected labels found: {sorted(uniq)}")


def add_split_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["site"] = df["site"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    df["split_key"] = df["site"] + "::" + df["group_id"]
    return df


def labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"].isin(VALID_LABELED)].copy()


def subset_by_keys(df: pd.DataFrame, keys: Set[str]) -> pd.DataFrame:
    if len(keys) == 0:
        return df.iloc[0:0].copy()
    return df[df["split_key"].isin(keys)].copy()


def subset_excluding_keys(df: pd.DataFrame, keys: Set[str]) -> pd.DataFrame:
    if len(keys) == 0:
        return df.copy()
    return df[~df["split_key"].isin(keys)].copy()


def summarize_df(df: pd.DataFrame) -> Dict:
    out = {
        "n_rows": int(len(df)),
        "n_groups": int(df["split_key"].nunique()) if "split_key" in df.columns else 0,
    }
    if "site" in df.columns:
        out["site_counts"] = {
            str(k): int(v) for k, v in df["site"].value_counts().to_dict().items()
        }
    if "label" in df.columns:
        out["label_counts"] = {
            str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
        }
    if "label_name" in df.columns:
        out["label_name_counts"] = {
            str(k): int(v) for k, v in df["label_name"].value_counts().to_dict().items()
        }
    return out

def debug_group_stats(df, site_name="unknown"):
    print("\n" + "=" * 80)
    print(f"[DEBUG] site = {site_name}")
    print(f"[DEBUG] total rows = {len(df)}")

    for cls in sorted(df["label_name"].unique()):
        sub = df[df["label_name"] == cls].copy()
        n_rows = len(sub)
        n_groups = sub["group_id"].nunique()
        print(f"[DEBUG] class={cls:10s} rows={n_rows:6d} groups={n_groups:4d}")

        g = sub.groupby("group_id").size().sort_values(ascending=False)
        print(f"        top10 group sizes: {g.head(10).tolist()}")

    print("=" * 80 + "\n")

# =========================================================
# validation helpers
# =========================================================
def split_label_counts(df: pd.DataFrame) -> Dict[int, int]:
    vc = df["label"].value_counts().to_dict()
    return {
        NOISE_LABEL: int(vc.get(NOISE_LABEL, 0)),
        EVENT_LABEL: int(vc.get(EVENT_LABEL, 0)),
    }


def is_valid_labeled_split_with_requirements(
    df: pd.DataFrame,
    required_counts: Dict[int, int],
    split_name: str = "split",
) -> Tuple[bool, str]:
    counts = split_label_counts(df)

    for label, required in required_counts.items():
        actual = counts.get(label, 0)
        if actual < required:
            label_name = {
                NOISE_LABEL: "noise",
                EVENT_LABEL: "event",
            }.get(label, str(label))
            return False, f"{split_name}: {label_name} count {actual} < required {required}"

    return True, "ok"


def validate_all_splits_with_requirements(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_required: Dict[int, int],
    val_required: Dict[int, int],
    test_required: Dict[int, int],
) -> Tuple[bool, List[str]]:
    msgs = []

    ok_train, msg_train = is_valid_labeled_split_with_requirements(
        train_df, train_required, "train"
    )
    ok_val, msg_val = is_valid_labeled_split_with_requirements(
        val_df, val_required, "val"
    )
    ok_test, msg_test = is_valid_labeled_split_with_requirements(
        test_df, test_required, "test"
    )

    msgs.extend([msg_train, msg_val, msg_test])
    return (ok_train and ok_val and ok_test), msgs


def assert_no_event_in_pretrain(pretrain_df: pd.DataFrame) -> None:
    n_event = int((pretrain_df["label"] == EVENT_LABEL).sum())
    if n_event > 0:
        raise RuntimeError(
            f"pretrain pool contains forbidden labeled event samples: {n_event}"
        )


# =========================================================
# group statistics
# =========================================================
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
    g["has_noise"] = g[NOISE_LABEL] > 0
    g["has_event"] = g[EVENT_LABEL] > 0
    return g[["split_key", NOISE_LABEL, EVENT_LABEL, "n_total", "has_noise", "has_event"]].copy()


# =========================================================
# generic balanced split core
# =========================================================
def commit_row_to_split(row: pd.Series, split_name: str, assigned: Dict[str, Dict]) -> None:
    g_noise = int(row[NOISE_LABEL])
    g_event = int(row[EVENT_LABEL])
    g_total = int(row["n_total"])

    assigned[split_name][NOISE_LABEL] += g_noise
    assigned[split_name][EVENT_LABEL] += g_event
    assigned[split_name]["n_total"] += g_total
    assigned[split_name]["keys"].append(str(row["split_key"]))

    if g_noise > 0:
        assigned[split_name]["noise_groups"] += 1
    if g_event > 0:
        assigned[split_name]["event_groups"] += 1


def assign_row_to_best_split(
    row: pd.Series,
    split_names: List[str],
    assigned: Dict[str, Dict],
    targets: Dict[str, Dict],
) -> str:
    g_noise = int(row[NOISE_LABEL])
    g_event = int(row[EVENT_LABEL])
    g_total = int(row["n_total"])

    best_split = None
    best_score = None

    for s in split_names:
        new_noise = assigned[s][NOISE_LABEL] + g_noise
        new_event = assigned[s][EVENT_LABEL] + g_event
        new_total = assigned[s]["n_total"] + g_total

        score = (
            abs(new_noise - targets[s][NOISE_LABEL]) / max(1.0, targets[s][NOISE_LABEL] + 1e-6)
            + abs(new_event - targets[s][EVENT_LABEL]) / max(1.0, targets[s][EVENT_LABEL] + 1e-6)
            + 0.5 * abs(new_total - targets[s]["n_total"]) / max(1.0, targets[s]["n_total"] + 1e-6)
        )

        fullness = assigned[s]["n_total"] / max(1.0, targets[s]["n_total"] + 1e-6)
        score += 0.25 * max(0.0, fullness - 1.0)

        if assigned[s]["event_groups"] == 0 and g_event > 0:
            score -= 0.75
        if assigned[s]["noise_groups"] == 0 and g_noise > 0:
            score -= 0.50

        if best_score is None or score < best_score:
            best_score = score
            best_split = s

    return best_split


def seed_required_groups(
    gdf: pd.DataFrame,
    split_names: List[str],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Dict], Set[str]]:
    assigned = {
        s: {
            NOISE_LABEL: 0,
            EVENT_LABEL: 0,
            "n_total": 0,
            "keys": [],
            "noise_groups": 0,
            "event_groups": 0,
        }
        for s in split_names
    }
    used_keys: Set[str] = set()

    event_rows = gdf[gdf[EVENT_LABEL] > 0].sample(
        frac=1.0,
        random_state=int(rng.integers(0, 1_000_000))
    )
    event_rows = event_rows.sort_values("n_total", ascending=False)

    if len(event_rows) < len(split_names):
        raise RuntimeError(
            f"Not enough event-containing groups to seed all splits: "
            f"{len(event_rows)} groups for {len(split_names)} splits"
        )

    for s, (_, row) in zip(split_names, event_rows.iterrows()):
        key = str(row["split_key"])
        if key in used_keys:
            continue
        commit_row_to_split(row, s, assigned)
        used_keys.add(key)

    noise_rows = gdf[gdf[NOISE_LABEL] > 0].sample(
        frac=1.0,
        random_state=int(rng.integers(0, 1_000_000))
    )
    noise_rows = noise_rows.sort_values("n_total", ascending=False)

    for s in split_names:
        if assigned[s]["noise_groups"] > 0:
            continue

        found = False
        for _, row in noise_rows.iterrows():
            key = str(row["split_key"])
            if key in used_keys:
                continue
            if int(row[NOISE_LABEL]) <= 0:
                continue
            commit_row_to_split(row, s, assigned)
            used_keys.add(key)
            found = True
            break

        if not found:
            raise RuntimeError(f"Could not find noise-containing group for split '{s}'")

    return assigned, used_keys


def balanced_group_split_once(
    df_labeled: pd.DataFrame,
    split_ratios: Dict[str, float],
    seed: int,
    required_counts_by_split: Dict[str, Dict[int, int]],
) -> Dict[str, Set[str]]:
    if not np.isclose(sum(split_ratios.values()), 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {split_ratios}")

    split_names = list(split_ratios.keys())
    gdf = group_label_table(df_labeled)
    rng = np.random.default_rng(seed)

    total_event_groups = int((gdf[EVENT_LABEL] > 0).sum())
    total_noise_groups = int((gdf[NOISE_LABEL] > 0).sum())

    if total_event_groups < len(split_names):
        raise RuntimeError(
            f"Impossible split: event-containing groups={total_event_groups}, "
            f"required splits={len(split_names)}"
        )
    if total_noise_groups < len(split_names):
        raise RuntimeError(
            f"Impossible split: noise-containing groups={total_noise_groups}, "
            f"required splits={len(split_names)}"
        )

    total_noise = int(gdf[NOISE_LABEL].sum())
    total_event = int(gdf[EVENT_LABEL].sum())
    total_all = int(gdf["n_total"].sum())

    targets = {
        s: {
            NOISE_LABEL: total_noise * split_ratios[s],
            EVENT_LABEL: total_event * split_ratios[s],
            "n_total": total_all * split_ratios[s],
        }
        for s in split_names
    }

    assigned, used_keys = seed_required_groups(gdf, split_names, rng)

    remain = gdf[~gdf["split_key"].isin(used_keys)].copy()
    remain = remain.sample(
        frac=1.0,
        random_state=int(rng.integers(0, 1_000_000))
    )
    remain = remain.sort_values("n_total", ascending=False).reset_index(drop=True)

    for _, row in remain.iterrows():
        best_split = assign_row_to_best_split(row, split_names, assigned, targets)
        commit_row_to_split(row, best_split, assigned)

    out = {s: set(assigned[s]["keys"]) for s in split_names}

    train_df = subset_by_keys(df_labeled, out.get("train", set()))
    val_df = subset_by_keys(df_labeled, out.get("val", set()))
    test_df = subset_by_keys(df_labeled, out.get("test", set()))

    ok, msgs = validate_all_splits_with_requirements(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_required=required_counts_by_split["train"],
        val_required=required_counts_by_split["val"],
        test_required=required_counts_by_split["test"],
    )
    if not ok:
        raise RuntimeError("Invalid split after assignment: " + " | ".join(msgs))

    return out


def balanced_group_split_with_retry(
    df_labeled: pd.DataFrame,
    split_ratios: Dict[str, float],
    seed: int,
    required_counts_by_split: Dict[str, Dict[int, int]],
    max_tries: int = 200,
) -> Tuple[Dict[str, Set[str]], int]:
    last_err = None

    for i in range(max_tries):
        cur_seed = int(seed + i)
        try:
            result = balanced_group_split_once(
                df_labeled=df_labeled,
                split_ratios=split_ratios,
                seed=cur_seed,
                required_counts_by_split=required_counts_by_split,
            )
            return result, cur_seed
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to create valid split after {max_tries} tries. Last error: {last_err}"
    )


def balanced_group_split_train_val_with_retry(
    df_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    train_required: Dict[int, int],
    val_required: Dict[int, int],
    max_tries: int = 200,
) -> Tuple[Set[str], Set[str], int]:
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError("train_ratio + val_ratio must be > 0")

    gdf = group_label_table(df_labeled)
    if int((gdf[EVENT_LABEL] > 0).sum()) < 2:
        raise RuntimeError("Need at least 2 event-containing groups for train/val split")
    if int((gdf[NOISE_LABEL] > 0).sum()) < 2:
        raise RuntimeError("Need at least 2 noise-containing groups for train/val split")

    last_err = None

    for i in range(max_tries):
        cur_seed = int(seed + i)
        try:
            rng = np.random.default_rng(cur_seed)
            split_names = ["train", "val"]

            total_noise = int(gdf[NOISE_LABEL].sum())
            total_event = int(gdf[EVENT_LABEL].sum())
            total_all = int(gdf["n_total"].sum())

            targets = {
                "train": {
                    NOISE_LABEL: total_noise * (train_ratio / total),
                    EVENT_LABEL: total_event * (train_ratio / total),
                    "n_total": total_all * (train_ratio / total),
                },
                "val": {
                    NOISE_LABEL: total_noise * (val_ratio / total),
                    EVENT_LABEL: total_event * (val_ratio / total),
                    "n_total": total_all * (val_ratio / total),
                },
            }

            assigned, used_keys = seed_required_groups(gdf, split_names, rng)

            remain = gdf[~gdf["split_key"].isin(used_keys)].copy()
            remain = remain.sample(
                frac=1.0,
                random_state=int(rng.integers(0, 1_000_000))
            )
            remain = remain.sort_values("n_total", ascending=False).reset_index(drop=True)

            for _, row in remain.iterrows():
                best_split = assign_row_to_best_split(row, split_names, assigned, targets)
                commit_row_to_split(row, best_split, assigned)

            train_keys = set(assigned["train"]["keys"])
            val_keys = set(assigned["val"]["keys"])

            train_df = subset_by_keys(df_labeled, train_keys)
            val_df = subset_by_keys(df_labeled, val_keys)

            ok_train, msg_train = is_valid_labeled_split_with_requirements(
                train_df, train_required, "train"
            )
            ok_val, msg_val = is_valid_labeled_split_with_requirements(
                val_df, val_required, "val"
            )

            if not (ok_train and ok_val):
                raise RuntimeError(msg_train + " | " + msg_val)

            return train_keys, val_keys, cur_seed

        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to create valid train/val split after {max_tries} tries. Last error: {last_err}"
    )


# =========================================================
# stage1 special split: event quota first
# =========================================================
def _sort_event_groups_desc(gdf_event: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if len(gdf_event) == 0:
        return gdf_event.copy()
    out = gdf_event.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).copy()
    out = out.sort_values([EVENT_LABEL, "n_total"], ascending=[False, False]).reset_index(drop=True)
    return out


def _sort_noise_groups_desc(gdf_noise: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if len(gdf_noise) == 0:
        return gdf_noise.copy()
    out = gdf_noise.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).copy()
    out = out.sort_values([NOISE_LABEL, "n_total"], ascending=[False, False]).reset_index(drop=True)
    return out


def _take_event_groups_until_quota(
    gdf_event: pd.DataFrame,
    quota_event: int,
    used_keys: Set[str],
    rng: np.random.Generator,
) -> Tuple[Set[str], int]:
    selected: Set[str] = set()
    event_sum = 0

    if quota_event <= 0:
        return selected, event_sum

    ordered = _sort_event_groups_desc(gdf_event[~gdf_event["split_key"].isin(used_keys)], rng)
    for _, row in ordered.iterrows():
        key = str(row["split_key"])
        if key in used_keys:
            continue
        selected.add(key)
        used_keys.add(key)
        event_sum += int(row[EVENT_LABEL])
        if event_sum >= quota_event:
            break

    return selected, event_sum


def _take_one_noise_group(
    gdf_noise: pd.DataFrame,
    used_keys: Set[str],
    rng: np.random.Generator,
) -> Optional[str]:
    ordered = _sort_noise_groups_desc(gdf_noise[~gdf_noise["split_key"].isin(used_keys)], rng)
    if len(ordered) == 0:
        return None
    row = ordered.iloc[0]
    key = str(row["split_key"])
    used_keys.add(key)
    return key


def _allocate_remaining_noise_groups(
    gdf_noise_remain: pd.DataFrame,
    split_names: List[str],
    split_keys: Dict[str, Set[str]],
    current_noise_counts: Dict[str, int],
    current_total_counts: Dict[str, int],
    targets_noise: Dict[str, float],
    targets_total: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, Set[str]]:
    remain = _sort_noise_groups_desc(gdf_noise_remain, rng)

    for _, row in remain.iterrows():
        key = str(row["split_key"])
        g_noise = int(row[NOISE_LABEL])
        g_total = int(row["n_total"])

        best_split = None
        best_score = None
        for s in split_names:
            new_noise = current_noise_counts[s] + g_noise
            new_total = current_total_counts[s] + g_total

            score = (
                abs(new_noise - targets_noise[s]) / max(1.0, targets_noise[s] + 1e-6)
                + 0.5 * abs(new_total - targets_total[s]) / max(1.0, targets_total[s] + 1e-6)
            )

            fullness = current_total_counts[s] / max(1.0, targets_total[s] + 1e-6)
            score += 0.2 * max(0.0, fullness - 1.0)

            if best_score is None or score < best_score:
                best_score = score
                best_split = s

        split_keys[best_split].add(key)
        current_noise_counts[best_split] += g_noise
        current_total_counts[best_split] += g_total

    return split_keys



def stage1_site_split_event_quota_first_once(
    df_site_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    train_required: Dict[int, int],
    val_required: Dict[int, int],
    test_required: Dict[int, int],
) -> Dict[str, Set[str]]:
    """
    Stage1 split with ratio-aware assignment.

    Previous implementation:
      - filled val/test minimum event quota
      - sent all remaining event groups to train
      - then tried to patch noise afterwards
    That avoided some failures but badly distorted train/val/test distribution
    for sites whose val/test minimum event requirement was small.

    New implementation:
      1) reserve minimum required event groups for each split
      2) reserve minimum required noise groups for each split (if still missing)
      3) assign ALL remaining groups (event / noise / mixed) with a ratio-aware scorer
      4) validate minimum requirements at the end

    This keeps group leakage-free splitting while making stage1 much closer to the
    requested train/val/test ratios.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = np.random.default_rng(seed)
    split_names = ["train", "val", "test"]

    gdf = group_label_table(df_site_labeled).copy()
    if len(gdf) == 0:
        raise RuntimeError("No labeled groups found for this site.")

    total_event_groups = int((gdf[EVENT_LABEL] > 0).sum())
    total_noise_groups = int((gdf[NOISE_LABEL] > 0).sum())
    if total_event_groups < len(split_names):
        raise RuntimeError(
            f"Impossible stage1 split: event-containing groups={total_event_groups}, "
            f"required splits={len(split_names)}"
        )
    if total_noise_groups < len(split_names):
        raise RuntimeError(
            f"Impossible stage1 split: noise-containing groups={total_noise_groups}, "
            f"required splits={len(split_names)}"
        )

    split_ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }

    total_noise = int(gdf[NOISE_LABEL].sum())
    total_event = int(gdf[EVENT_LABEL].sum())
    total_all = int(gdf["n_total"].sum())

    required_counts_by_split = {
        "train": {
            NOISE_LABEL: int(train_required.get(NOISE_LABEL, 0)),
            EVENT_LABEL: int(train_required.get(EVENT_LABEL, 0)),
        },
        "val": {
            NOISE_LABEL: int(val_required.get(NOISE_LABEL, 0)),
            EVENT_LABEL: int(val_required.get(EVENT_LABEL, 0)),
        },
        "test": {
            NOISE_LABEL: int(test_required.get(NOISE_LABEL, 0)),
            EVENT_LABEL: int(test_required.get(EVENT_LABEL, 0)),
        },
    }

    required_total_event = sum(v[EVENT_LABEL] for v in required_counts_by_split.values())
    required_total_noise = sum(v[NOISE_LABEL] for v in required_counts_by_split.values())
    if total_event < required_total_event:
        raise RuntimeError(
            f"Not enough total event samples: total_event={total_event}, required={required_total_event}"
        )
    if total_noise < required_total_noise:
        raise RuntimeError(
            f"Not enough total noise samples: total_noise={total_noise}, required={required_total_noise}"
        )

    targets = {
        s: {
            NOISE_LABEL: total_noise * split_ratios[s],
            EVENT_LABEL: total_event * split_ratios[s],
            "n_total": total_all * split_ratios[s],
        }
        for s in split_names
    }

    assigned = {
        s: {
            NOISE_LABEL: 0,
            EVENT_LABEL: 0,
            "n_total": 0,
            "keys": [],
            "noise_groups": 0,
            "event_groups": 0,
        }
        for s in split_names
    }
    used_keys: Set[str] = set()

    def _commit_key_to_split(split_name: str, row: pd.Series) -> None:
        key = str(row["split_key"])
        if key in used_keys:
            return
        commit_row_to_split(row, split_name, assigned)
        used_keys.add(key)

    def _reserve_min_requirement(label: int, split_name: str, quota: int) -> None:
        if quota <= 0:
            return

        current = assigned[split_name][label]
        if current >= quota:
            return

        candidates = gdf[(gdf["split_key"].isin(set(gdf["split_key"]) - used_keys)) & (gdf[label] > 0)].copy()
        if len(candidates) == 0:
            lname = "event" if label == EVENT_LABEL else "noise"
            raise RuntimeError(f"No remaining {lname}-containing groups for split '{split_name}'")

        # small randomization first
        candidates = candidates.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 1_000_000))
        ).reset_index(drop=True)

        # Prefer groups that help quota without excessive overshoot.
        deficit = quota - current

        def _score_row(row: pd.Series) -> tuple:
            label_count = int(row[label])
            new_label = current + label_count
            new_total = assigned[split_name]["n_total"] + int(row["n_total"])
            overshoot = max(0, new_label - quota)
            target_gap = abs(new_total - targets[split_name]["n_total"])
            other_label = NOISE_LABEL if label == EVENT_LABEL else EVENT_LABEL
            other_bonus = -int(row[other_label] > 0)  # prefer mixed groups slightly
            exact_gap = abs(deficit - label_count)
            return (overshoot, exact_gap, target_gap, other_bonus, -label_count, -int(row["n_total"]))

        while assigned[split_name][label] < quota:
            candidates = candidates[~candidates["split_key"].isin(used_keys)].copy()
            candidates = candidates[candidates[label] > 0].copy()
            if len(candidates) == 0:
                lname = "event" if label == EVENT_LABEL else "noise"
                raise RuntimeError(
                    f"Could not satisfy {lname} quota for split '{split_name}': "
                    f"current={assigned[split_name][label]}, required={quota}"
                )
            best_idx = min(range(len(candidates)), key=lambda i: _score_row(candidates.iloc[i]))
            row = candidates.iloc[best_idx]
            _commit_key_to_split(split_name, row)

    # ------------------------------------------------------------------
    # 1) reserve minimum event requirement first, but in ratio order:
    #    smaller target splits first (usually val/test before train)
    # ------------------------------------------------------------------
    for s in sorted(split_names, key=lambda x: targets[x][EVENT_LABEL]):
        _reserve_min_requirement(EVENT_LABEL, s, required_counts_by_split[s][EVENT_LABEL])

    # ------------------------------------------------------------------
    # 2) reserve minimum noise requirement for splits still lacking noise
    # ------------------------------------------------------------------
    for s in sorted(split_names, key=lambda x: targets[x][NOISE_LABEL]):
        _reserve_min_requirement(NOISE_LABEL, s, required_counts_by_split[s][NOISE_LABEL])

    # ------------------------------------------------------------------
    # 3) assign all remaining groups with ratio-aware scoring
    # ------------------------------------------------------------------
    remain = gdf[~gdf["split_key"].isin(used_keys)].copy()
    remain = remain.sample(
        frac=1.0,
        random_state=int(rng.integers(0, 1_000_000))
    )
    remain = remain.sort_values(
        [EVENT_LABEL, NOISE_LABEL, "n_total"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    for _, row in remain.iterrows():
        best_split = assign_row_to_best_split(row, split_names, assigned, targets)
        commit_row_to_split(row, best_split, assigned)

    split_keys = {s: set(assigned[s]["keys"]) for s in split_names}

    train_df = subset_by_keys(df_site_labeled, split_keys["train"])
    val_df = subset_by_keys(df_site_labeled, split_keys["val"])
    test_df = subset_by_keys(df_site_labeled, split_keys["test"])

    ok, msgs = validate_all_splits_with_requirements(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_required=train_required,
        val_required=val_required,
        test_required=test_required,
    )
    if not ok:
        debug_info = {
            "totals": {
                "noise": total_noise,
                "event": total_event,
                "all": total_all,
            },
            "targets": {
                s: {
                    "noise": float(targets[s][NOISE_LABEL]),
                    "event": float(targets[s][EVENT_LABEL]),
                    "all": float(targets[s]["n_total"]),
                } for s in split_names
            },
            "assigned": {
                s: {
                    "noise": int(assigned[s][NOISE_LABEL]),
                    "event": int(assigned[s][EVENT_LABEL]),
                    "all": int(assigned[s]["n_total"]),
                    "n_groups": int(len(assigned[s]["keys"])),
                } for s in split_names
            },
        }
        raise RuntimeError(
            "Invalid split after ratio-aware stage1 assignment: "
            + " | ".join(msgs)
            + f" | debug={debug_info}"
        )

    return split_keys


def stage1_site_split_event_quota_first_with_retry(
    df_site_labeled: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    train_required: Dict[int, int],
    val_required: Dict[int, int],
    test_required: Dict[int, int],
    max_tries: int,
) -> Tuple[Dict[str, Set[str]], int]:
    last_err = None
    for i in range(max_tries):
        cur_seed = int(seed + i)
        try:
            split_keys = stage1_site_split_event_quota_first_once(
                df_site_labeled=df_site_labeled,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=cur_seed,
                train_required=train_required,
                val_required=val_required,
                test_required=test_required,
            )
            return split_keys, cur_seed
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to create valid stage1 site-only split after {max_tries} tries. Last error: {last_err}"
    )


# =========================================================
# pretrain policy
# =========================================================
def cap_unlabeled_per_site(
    df: pd.DataFrame,
    max_unlabel_per_site: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if max_unlabel_per_site is None or max_unlabel_per_site <= 0:
        return df.copy()

    labeled = df[df["label"] != UNLABEL_LABEL].copy()
    unlabel = df[df["label"] == UNLABEL_LABEL].copy()

    rng = np.random.default_rng(seed)
    kept = []

    for site, g in unlabel.groupby("site", sort=False):
        if len(g) <= max_unlabel_per_site:
            kept.append(g.copy())
        else:
            idx = rng.choice(len(g), size=max_unlabel_per_site, replace=False)
            kept.append(g.iloc[idx].copy())

    kept_unlabel_df = pd.concat(kept, ignore_index=True) if kept else unlabel.iloc[0:0].copy()
    out = pd.concat([labeled, kept_unlabel_df], ignore_index=True)
    return out.reset_index(drop=True)


def build_pretrain_pool_no_event(
    df: pd.DataFrame,
    exclude_keys: Set[str],
    max_unlabel_per_site: Optional[int],
    seed: int,
) -> pd.DataFrame:
    pool = subset_excluding_keys(df, exclude_keys).copy()

    # labeled event is forbidden in pretrain
    pool = pool[pool["label"] != EVENT_LABEL].copy()

    pool = cap_unlabeled_per_site(
        pool,
        max_unlabel_per_site=max_unlabel_per_site,
        seed=seed,
    )

    assert_no_event_in_pretrain(pool)
    return pool.reset_index(drop=True)


# =========================================================
# save helper
# =========================================================
def save_bundle(
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


# =========================================================
# requirement helpers
# =========================================================
def make_stage1_requirements(
    site_name: str,
    min_train_noise: int,
    min_train_event: int,
    min_val_noise: int,
    min_val_event: int,
    min_test_noise: int,
    min_test_event: int,
    min_val_event_utah2019: Optional[int],
    min_test_event_utah2019: Optional[int],
) -> Dict[str, Dict[int, int]]:
    val_event_req = min_val_event
    test_event_req = min_test_event

    if site_name == "utah_2019":
        if min_val_event_utah2019 is not None:
            val_event_req = min_val_event_utah2019
        if min_test_event_utah2019 is not None:
            test_event_req = min_test_event_utah2019

    return {
        "train": {
            NOISE_LABEL: int(min_train_noise),
            EVENT_LABEL: int(min_train_event),
        },
        "val": {
            NOISE_LABEL: int(min_val_noise),
            EVENT_LABEL: int(val_event_req),
        },
        "test": {
            NOISE_LABEL: int(min_test_noise),
            EVENT_LABEL: int(test_event_req),
        },
    }


def make_train_val_requirements(
    min_train_noise: int,
    min_train_event: int,
    min_val_noise: int,
    min_val_event: int,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    train_required = {
        NOISE_LABEL: int(min_train_noise),
        EVENT_LABEL: int(min_train_event),
    }
    val_required = {
        NOISE_LABEL: int(min_val_noise),
        EVENT_LABEL: int(min_val_event),
    }
    return train_required, val_required


# =========================================================
# experiment builders
# =========================================================
def build_stage1_site_only(
    df_all: pd.DataFrame,
    site_name: str,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_unlabel_per_site: Optional[int],
    max_split_tries: int,
    min_train_noise: int,
    min_train_event: int,
    min_val_noise: int,
    min_val_event: int,
    min_test_noise: int,
    min_test_event: int,
    min_val_event_utah2019: Optional[int],
    min_test_event_utah2019: Optional[int],
) -> None:
    exp_name = f"stage1_{site_name}_only"
    out_dir = out_root / exp_name

    df_site = df_all[df_all["site"] == site_name].copy()
    df_site_labeled = labeled_only(df_site)

    required_counts_by_split = make_stage1_requirements(
        site_name=site_name,
        min_train_noise=min_train_noise,
        min_train_event=min_train_event,
        min_val_noise=min_val_noise,
        min_val_event=min_val_event,
        min_test_noise=min_test_noise,
        min_test_event=min_test_event,
        min_val_event_utah2019=min_val_event_utah2019,
        min_test_event_utah2019=min_test_event_utah2019,
    )

    split_keys, used_seed = stage1_site_split_event_quota_first_with_retry(
        df_site_labeled=df_site_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        train_required=required_counts_by_split["train"],
        val_required=required_counts_by_split["val"],
        test_required=required_counts_by_split["test"],
        max_tries=max_split_tries,
    )

    train_df = subset_by_keys(df_site_labeled, split_keys["train"])
    val_df = subset_by_keys(df_site_labeled, split_keys["val"])
    test_df = subset_by_keys(df_site_labeled, split_keys["test"])

    pretrain_df = build_pretrain_pool_no_event(
        df=df_site,
        exclude_keys=split_keys["test"],
        max_unlabel_per_site=max_unlabel_per_site,
        seed=used_seed,
    )

    ok, msgs = validate_all_splits_with_requirements(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_required=required_counts_by_split["train"],
        val_required=required_counts_by_split["val"],
        test_required=required_counts_by_split["test"],
    )

    summary = {
        "experiment": exp_name,
        "requested_seed": int(seed),
        "used_seed": int(used_seed),
        "site": site_name,
        "description": (
            "Site-specific experiment. Pretrain on same-site non-test pool using only "
            "noise+unlabeled samples (labeled event excluded). "
            "Fine-tune/evaluate on same-site labeled splits. "
            "Stage1 uses event-quota-first allocation, then noise allocation."
        ),
        "constraints": {
            "group_leakage_free": True,
            "max_split_tries": int(max_split_tries),
            "pretrain_forbid_labeled_event": True,
            "stage1_event_quota_first": True,
            "required_counts_by_split": {
                k: {str(lbl): int(v) for lbl, v in req.items()}
                for k, req in required_counts_by_split.items()
            },
        },
        "validation": {
            "ok": bool(ok),
            "messages": msgs,
        },
        "pretrain": summarize_df(pretrain_df),
        "train": summarize_df(train_df),
        "val": summarize_df(val_df),
        "test": summarize_df(test_df),
    }
    save_bundle(out_dir, pretrain_df, train_df, val_df, test_df, summary)


def build_stage2_joint_all(
    df_all: pd.DataFrame,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_unlabel_per_site: Optional[int],
    max_split_tries: int,
    min_train_noise: int,
    min_train_event: int,
    min_val_noise: int,
    min_val_event: int,
    min_test_noise: int,
    min_test_event: int,
) -> None:
    exp_name = "stage2_joint_all"
    out_dir = out_root / exp_name

    df_labeled = labeled_only(df_all)

    split_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    required_counts_by_split = {
        "train": {
            NOISE_LABEL: int(min_train_noise),
            EVENT_LABEL: int(min_train_event),
        },
        "val": {
            NOISE_LABEL: int(min_val_noise),
            EVENT_LABEL: int(min_val_event),
        },
        "test": {
            NOISE_LABEL: int(min_test_noise),
            EVENT_LABEL: int(min_test_event),
        },
    }

    split_keys, used_seed = balanced_group_split_with_retry(
        df_labeled=df_labeled,
        split_ratios=split_ratios,
        seed=seed,
        required_counts_by_split=required_counts_by_split,
        max_tries=max_split_tries,
    )

    train_df = subset_by_keys(df_labeled, split_keys["train"])
    val_df = subset_by_keys(df_labeled, split_keys["val"])
    test_df = subset_by_keys(df_labeled, split_keys["test"])

    pretrain_df = build_pretrain_pool_no_event(
        df=df_all,
        exclude_keys=split_keys["test"],
        max_unlabel_per_site=max_unlabel_per_site,
        seed=used_seed,
    )

    ok, msgs = validate_all_splits_with_requirements(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_required=required_counts_by_split["train"],
        val_required=required_counts_by_split["val"],
        test_required=required_counts_by_split["test"],
    )

    summary = {
        "experiment": exp_name,
        "requested_seed": int(seed),
        "used_seed": int(used_seed),
        "description": (
            "Joint multi-site experiment. Pretrain on all non-test pool using only "
            "noise+unlabeled samples (labeled event excluded). "
            "Train/val/test on labeled only. "
            "Generic balanced group split with minimum count constraints."
        ),
        "constraints": {
            "group_leakage_free": True,
            "max_split_tries": int(max_split_tries),
            "pretrain_forbid_labeled_event": True,
            "required_counts_by_split": {
                k: {str(lbl): int(v) for lbl, v in req.items()}
                for k, req in required_counts_by_split.items()
            },
        },
        "validation": {
            "ok": bool(ok),
            "messages": msgs,
        },
        "pretrain": summarize_df(pretrain_df),
        "train": summarize_df(train_df),
        "val": summarize_df(val_df),
        "test": summarize_df(test_df),
    }
    save_bundle(out_dir, pretrain_df, train_df, val_df, test_df, summary)


def build_stage3_pairwise(
    df_all: pd.DataFrame,
    src_site: str,
    tgt_site: str,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_unlabel_per_site: Optional[int],
    max_split_tries: int,
    min_train_noise: int,
    min_train_event: int,
    min_val_noise: int,
    min_val_event: int,
    min_test_noise: int,
    min_test_event: int,
) -> None:
    exp_name = f"stage3_{src_site}_to_{tgt_site}"
    out_dir = out_root / exp_name

    df_src = df_all[df_all["site"] == src_site].copy()
    df_tgt = df_all[df_all["site"] == tgt_site].copy()

    df_src_labeled = labeled_only(df_src)
    df_tgt_labeled = labeled_only(df_tgt)

    train_required, val_required = make_train_val_requirements(
        min_train_noise=min_train_noise,
        min_train_event=min_train_event,
        min_val_noise=min_val_noise,
        min_val_event=min_val_event,
    )

    train_keys, val_keys, used_seed = balanced_group_split_train_val_with_retry(
        df_labeled=df_src_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        train_required=train_required,
        val_required=val_required,
        max_tries=max_split_tries,
    )

    train_df = subset_by_keys(df_src_labeled, train_keys)
    val_df = subset_by_keys(df_src_labeled, val_keys)
    test_df = df_tgt_labeled.copy()

    test_required = {
        NOISE_LABEL: int(min_test_noise),
        EVENT_LABEL: int(min_test_event),
    }
    ok_test, msg_test = is_valid_labeled_split_with_requirements(
        test_df, test_required, "test(target)"
    )
    if not ok_test:
        raise RuntimeError(f"[{exp_name}] target site labeled set invalid: {msg_test}")

    pretrain_df = build_pretrain_pool_no_event(
        df=df_src,
        exclude_keys=set(),
        max_unlabel_per_site=max_unlabel_per_site,
        seed=used_seed,
    )

    ok_train, msg_train = is_valid_labeled_split_with_requirements(train_df, train_required, "train")
    ok_val, msg_val = is_valid_labeled_split_with_requirements(val_df, val_required, "val")

    summary = {
        "experiment": exp_name,
        "requested_seed": int(seed),
        "used_seed": int(used_seed),
        "source_site": src_site,
        "target_site": tgt_site,
        "description": (
            "Cross-site transfer. Train/val on source labeled only, test on target labeled only. "
            "Pretrain on source pool using only noise+unlabeled samples (labeled event excluded). "
            "Source split is leakage-safe."
        ),
        "constraints": {
            "group_leakage_free": True,
            "max_split_tries": int(max_split_tries),
            "pretrain_forbid_labeled_event": True,
            "train_required": {str(k): int(v) for k, v in train_required.items()},
            "val_required": {str(k): int(v) for k, v in val_required.items()},
            "test_required": {str(k): int(v) for k, v in test_required.items()},
        },
        "validation": {
            "ok": bool(ok_train and ok_val and ok_test),
            "messages": [msg_train, msg_val, msg_test],
        },
        "pretrain": summarize_df(pretrain_df),
        "train": summarize_df(train_df),
        "val": summarize_df(val_df),
        "test": summarize_df(test_df),
    }
    save_bundle(out_dir, pretrain_df, train_df, val_df, test_df, summary)


def build_stage4_leave_one_out(
    df_all: pd.DataFrame,
    heldout_site: str,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_unlabel_per_site: Optional[int],
    max_split_tries: int,
    min_train_noise: int,
    min_train_event: int,
    min_val_noise: int,
    min_val_event: int,
    min_test_noise: int,
    min_test_event: int,
) -> None:
    exp_name = f"stage4_leave_one_site_out_{heldout_site}"
    out_dir = out_root / exp_name

    train_sites = sorted(set(df_all["site"]) - {heldout_site})
    df_src = df_all[df_all["site"].isin(train_sites)].copy()
    df_tgt = df_all[df_all["site"] == heldout_site].copy()

    df_src_labeled = labeled_only(df_src)
    df_tgt_labeled = labeled_only(df_tgt)

    train_required, val_required = make_train_val_requirements(
        min_train_noise=min_train_noise,
        min_train_event=min_train_event,
        min_val_noise=min_val_noise,
        min_val_event=min_val_event,
    )

    train_keys, val_keys, used_seed = balanced_group_split_train_val_with_retry(
        df_labeled=df_src_labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        train_required=train_required,
        val_required=val_required,
        max_tries=max_split_tries,
    )

    train_df = subset_by_keys(df_src_labeled, train_keys)
    val_df = subset_by_keys(df_src_labeled, val_keys)
    test_df = df_tgt_labeled.copy()

    test_required = {
        NOISE_LABEL: int(min_test_noise),
        EVENT_LABEL: int(min_test_event),
    }
    ok_test, msg_test = is_valid_labeled_split_with_requirements(
        test_df, test_required, "test(heldout)"
    )
    if not ok_test:
        raise RuntimeError(f"[{exp_name}] heldout site labeled set invalid: {msg_test}")

    pretrain_df = build_pretrain_pool_no_event(
        df=df_src,
        exclude_keys=set(),
        max_unlabel_per_site=max_unlabel_per_site,
        seed=used_seed,
    )

    ok_train, msg_train = is_valid_labeled_split_with_requirements(train_df, train_required, "train")
    ok_val, msg_val = is_valid_labeled_split_with_requirements(val_df, val_required, "val")

    summary = {
        "experiment": exp_name,
        "requested_seed": int(seed),
        "used_seed": int(used_seed),
        "heldout_site": heldout_site,
        "source_sites": train_sites,
        "description": (
            "Leave-one-site-out DG. Train/val on source sites, test on held-out site. "
            "Pretrain on source pool using only noise+unlabeled samples (labeled event excluded). "
            "Source split is leakage-safe."
        ),
        "constraints": {
            "group_leakage_free": True,
            "max_split_tries": int(max_split_tries),
            "pretrain_forbid_labeled_event": True,
            "train_required": {str(k): int(v) for k, v in train_required.items()},
            "val_required": {str(k): int(v) for k, v in val_required.items()},
            "test_required": {str(k): int(v) for k, v in test_required.items()},
        },
        "validation": {
            "ok": bool(ok_train and ok_val and ok_test),
            "messages": [msg_train, msg_val, msg_test],
        },
        "pretrain": summarize_df(pretrain_df),
        "train": summarize_df(train_df),
        "val": summarize_df(val_df),
        "test": summarize_df(test_df),
    }
    save_bundle(out_dir, pretrain_df, train_df, val_df, test_df, summary)


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_csv", required=True, help="all_samples.csv from FINAL build")
    ap.add_argument("--out_dir", required=True, help="experiments output root")
    ap.add_argument("--sites", nargs="+", default=["pohang", "utah_2019", "utah_2023"])

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_split_tries", type=int, default=1000)

    ap.add_argument(
        "--max_unlabel_per_site",
        type=int,
        default=3000,
        help="cap unlabeled samples per site in pretrain.csv; <=0 means use all",
    )

    ap.add_argument("--min_train_noise", type=int, default=1)
    ap.add_argument("--min_train_event", type=int, default=1)
    ap.add_argument("--min_val_noise", type=int, default=1)
    ap.add_argument("--min_val_event", type=int, default=1)
    ap.add_argument("--min_test_noise", type=int, default=1)
    ap.add_argument("--min_test_event", type=int, default=1)

    ap.add_argument("--min_val_event_utah2019", type=int, default=30)
    ap.add_argument("--min_test_event_utah2019", type=int, default=30)

    args = ap.parse_args()

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    df = pd.read_csv(args.all_csv)
    validate_df(df)
    df = add_split_key(df)

    max_unlabel_per_site = args.max_unlabel_per_site if args.max_unlabel_per_site > 0 else None

    requested_sites = [str(x) for x in args.sites]
    available_sites = [str(x) for x in sorted(pd.unique(df["site"]))]
    active_sites = [site for site in requested_sites if site in available_sites]
    skipped_sites = [site for site in requested_sites if site not in available_sites]

    if len(active_sites) == 0:
        raise RuntimeError(
            f"None of the requested sites are present in all_csv. "
            f"requested={requested_sites}, available={available_sites}"
        )

    site_selection = {
        "requested_sites": requested_sites,
        "available_sites": available_sites,
        "active_sites": active_sites,
        "skipped_sites": skipped_sites,
    }
    with open(out_root / "site_selection.json", "w", encoding="utf-8") as f:
        json.dump(site_selection, f, ensure_ascii=False, indent=2)

    if skipped_sites:
        print(f"[WARN] skipping absent sites: {skipped_sites}; available={available_sites}")

    # stage 1
    for site in active_sites:
        build_stage1_site_only(
            df_all=df,
            site_name=site,
            out_root=out_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            max_unlabel_per_site=max_unlabel_per_site,
            max_split_tries=args.max_split_tries,
            min_train_noise=args.min_train_noise,
            min_train_event=args.min_train_event,
            min_val_noise=args.min_val_noise,
            min_val_event=args.min_val_event,
            min_test_noise=args.min_test_noise,
            min_test_event=args.min_test_event,
            min_val_event_utah2019=args.min_val_event_utah2019,
            min_test_event_utah2019=args.min_test_event_utah2019,
        )

    # stage 2
    build_stage2_joint_all(
        df_all=df,
        out_root=out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_unlabel_per_site=max_unlabel_per_site,
        max_split_tries=args.max_split_tries,
        min_train_noise=args.min_train_noise,
        min_train_event=args.min_train_event,
        min_val_noise=args.min_val_noise,
        min_val_event=args.min_val_event,
        min_test_noise=args.min_test_noise,
        min_test_event=args.min_test_event,
    )

    # stage 3
    if len(active_sites) >= 2:
        for src in active_sites:
            for tgt in active_sites:
                if src == tgt:
                    continue
                build_stage3_pairwise(
                    df_all=df,
                    src_site=src,
                    tgt_site=tgt,
                    out_root=out_root,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                    max_unlabel_per_site=max_unlabel_per_site,
                    max_split_tries=args.max_split_tries,
                    min_train_noise=args.min_train_noise,
                    min_train_event=args.min_train_event,
                    min_val_noise=args.min_val_noise,
                    min_val_event=args.min_val_event,
                    min_test_noise=args.min_test_noise,
                    min_test_event=args.min_test_event,
                )
    else:
        print("[WARN] skipping stage3 pairwise splits because fewer than 2 active sites are available")

    # stage 4
    if len(active_sites) >= 2:
        for heldout in active_sites:
            build_stage4_leave_one_out(
                df_all=df,
                heldout_site=heldout,
                out_root=out_root,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed,
                max_unlabel_per_site=max_unlabel_per_site,
                max_split_tries=args.max_split_tries,
                min_train_noise=args.min_train_noise,
                min_train_event=args.min_train_event,
                min_val_noise=args.min_val_noise,
                min_val_event=args.min_val_event,
                min_test_noise=args.min_test_noise,
                min_test_event=args.min_test_event,
            )
    else:
        print("[WARN] skipping stage4 leave-one-site-out splits because fewer than 2 active sites are available")

    print(f"[DONE] saved experiments under: {out_root}")


if __name__ == "__main__":
    main()