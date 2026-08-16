#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

NOISE_LABEL = 0
EVENT_LABEL = 1
UNLABEL_LABEL = 2
VALID_LABELED = {NOISE_LABEL, EVENT_LABEL}
VALID_ALL = {NOISE_LABEL, EVENT_LABEL, UNLABEL_LABEL}
SPLITS = ["train", "val", "test"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_df(df: pd.DataFrame) -> None:
    required = ["site", "label", "group_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    labels = set(pd.unique(df["label"]))
    if not labels.issubset(VALID_ALL):
        raise ValueError(f"Unexpected labels: {sorted(labels)}")


def add_split_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["site"] = out["site"].astype(str)
    out["group_id"] = out["group_id"].astype(str)
    out["split_key"] = out["site"] + "::" + out["group_id"]
    return out


def labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"].isin(VALID_LABELED)].copy()


def subset_by_keys(df: pd.DataFrame, keys: Set[str]) -> pd.DataFrame:
    if not keys:
        return df.iloc[0:0].copy()
    return df[df["split_key"].isin(keys)].copy()


def subset_excluding_keys(df: pd.DataFrame, keys: Set[str]) -> pd.DataFrame:
    if not keys:
        return df.copy()
    return df[~df["split_key"].isin(keys)].copy()


def summarize_df(df: pd.DataFrame) -> Dict:
    out = {
        "n_rows": int(len(df)),
        "n_groups": int(df["split_key"].nunique()) if "split_key" in df.columns else 0,
    }
    if "site" in df.columns:
        out["site_counts"] = {str(k): int(v) for k, v in df["site"].value_counts().sort_index().items()}
    if "label" in df.columns:
        out["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
        if "site" in df.columns:
            out["site_label_counts"] = {
                f"{site}|{label}": int(v)
                for (site, label), v in df.groupby(["site", "label"]).size().sort_index().items()
            }
    return out


def group_table(df_labeled: pd.DataFrame) -> pd.DataFrame:
    g = df_labeled.groupby(["split_key", "label"]).size().unstack(fill_value=0).reset_index()
    for label in [NOISE_LABEL, EVENT_LABEL]:
        if label not in g.columns:
            g[label] = 0
    g["n_total"] = g[NOISE_LABEL] + g[EVENT_LABEL]
    return g[["split_key", NOISE_LABEL, EVENT_LABEL, "n_total"]].copy()


def ratio_targets(gdf: pd.DataFrame, ratios: Dict[str, float]) -> Dict[str, Dict[int | str, float]]:
    total_noise = float(gdf[NOISE_LABEL].sum())
    total_event = float(gdf[EVENT_LABEL].sum())
    total_all = float(gdf["n_total"].sum())
    return {
        split: {
            NOISE_LABEL: total_noise * ratio,
            EVENT_LABEL: total_event * ratio,
            "n_total": total_all * ratio,
        }
        for split, ratio in ratios.items()
    }


def empty_assigned() -> Dict[str, Dict]:
    return {
        s: {NOISE_LABEL: 0, EVENT_LABEL: 0, "n_total": 0, "keys": []}
        for s in SPLITS
    }


def add_row(assigned: Dict[str, Dict], split: str, row: pd.Series) -> None:
    assigned[split][NOISE_LABEL] += int(row[NOISE_LABEL])
    assigned[split][EVENT_LABEL] += int(row[EVENT_LABEL])
    assigned[split]["n_total"] += int(row["n_total"])
    assigned[split]["keys"].append(str(row["split_key"]))


def assignment_objective(
    assigned: Dict[str, Dict],
    targets: Dict[str, Dict[int | str, float]],
    min_counts: Dict[str, Dict[int, int]],
) -> float:
    score = 0.0
    for split in SPLITS:
        for key, weight in [(NOISE_LABEL, 2.0), (EVENT_LABEL, 2.0), ("n_total", 0.7)]:
            target = float(targets[split][key])
            actual = float(assigned[split][key])
            score += weight * ((actual - target) / max(1.0, target)) ** 2
        n = assigned[split][NOISE_LABEL]
        e = assigned[split][EVENT_LABEL]
        total = max(1, n + e)
        target_total = max(1.0, targets[split][NOISE_LABEL] + targets[split][EVENT_LABEL])
        target_event_frac = targets[split][EVENT_LABEL] / target_total
        score += 2.0 * ((e / total) - target_event_frac) ** 2
        for label, required in min_counts.get(split, {}).items():
            deficit = max(0, int(required) - int(assigned[split][label]))
            score += 1000.0 * deficit * deficit
    return float(score)


def assign_greedy(gdf: pd.DataFrame, targets: Dict[str, Dict], min_counts: Dict[str, Dict[int, int]], seed: int) -> Dict[str, Dict]:
    rng = np.random.default_rng(seed)
    assigned = empty_assigned()
    rows = gdf.copy()
    rows["importance"] = rows[[NOISE_LABEL, EVENT_LABEL]].max(axis=1) + 0.25 * rows["n_total"]
    rows = rows.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
    records = rows.to_dict("records")
    records.sort(key=lambda r: float(r["importance"]), reverse=True)

    def clone_assigned() -> Dict[str, Dict]:
        return {
            s: {
                NOISE_LABEL: assigned[s][NOISE_LABEL],
                EVENT_LABEL: assigned[s][EVENT_LABEL],
                "n_total": assigned[s]["n_total"],
                "keys": list(assigned[s]["keys"]),
            }
            for s in SPLITS
        }

    for row in records:
        best_split = None
        best_score = None
        for split in SPLITS:
            trial = clone_assigned()
            add_row(trial, split, row)
            score = assignment_objective(trial, targets, min_counts)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        add_row(assigned, best_split, row)
    return assigned


def assigned_from_mapping(gdf: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Dict]:
    assigned = empty_assigned()
    by_key = {str(r["split_key"]): r for _, r in gdf.iterrows()}
    for key, split in mapping.items():
        add_row(assigned, split, by_key[key])
    return assigned


def improve_by_moves(
    gdf: pd.DataFrame,
    assigned: Dict[str, Dict],
    targets: Dict[str, Dict],
    min_counts: Dict[str, Dict[int, int]],
    max_passes: int = 3,
) -> Dict[str, Dict]:
    rows = {str(r["split_key"]): r for _, r in gdf.iterrows()}
    mapping = {key: split for split, vals in assigned.items() for key in vals["keys"]}
    best_score = assignment_objective(assigned, targets, min_counts)

    for _ in range(max_passes):
        improved = False
        keys = list(mapping.keys())
        for key in keys:
            old_split = mapping[key]
            for new_split in SPLITS:
                if new_split == old_split:
                    continue
                trial_mapping = dict(mapping)
                trial_mapping[key] = new_split
                trial = assigned_from_mapping(gdf, trial_mapping)
                score = assignment_objective(trial, targets, min_counts)
                if score + 1e-12 < best_score:
                    mapping = trial_mapping
                    assigned = trial
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return assigned


def make_site_split(
    df_site: pd.DataFrame,
    ratios: Dict[str, float],
    seed: int,
    max_tries: int,
    min_counts: Dict[str, Dict[int, int]],
) -> Tuple[Dict[str, Set[str]], int, Dict]:
    gdf = group_table(labeled_only(df_site))
    if len(gdf) == 0:
        raise ValueError("No labeled groups")
    targets = ratio_targets(gdf, ratios)

    best = None
    best_seed = seed
    best_score = math.inf
    for i in range(max_tries):
        cur_seed = seed + i
        assigned = assign_greedy(gdf, targets, min_counts, cur_seed)
        score = assignment_objective(assigned, targets, min_counts)
        if score < best_score:
            best = assigned
            best_seed = cur_seed
            best_score = score
            if score < 0.02:
                break

    if best is None:
        raise RuntimeError("No split candidate generated")

    split_keys = {s: set(best[s]["keys"]) for s in SPLITS}
    stats = {
        "used_seed": int(best_seed),
        "objective": float(best_score),
        "targets": {
            s: {"noise": float(targets[s][NOISE_LABEL]), "event": float(targets[s][EVENT_LABEL]), "rows": float(targets[s]["n_total"])}
            for s in SPLITS
        },
        "assigned": {
            s: {"noise": int(best[s][NOISE_LABEL]), "event": int(best[s][EVENT_LABEL]), "rows": int(best[s]["n_total"]), "groups": len(best[s]["keys"])}
            for s in SPLITS
        },
        "min_counts": {s: {str(k): int(v) for k, v in req.items()} for s, req in min_counts.items()},
    }
    return split_keys, best_seed, stats


def build_pretrain(df: pd.DataFrame, exclude_keys: Set[str], max_unlabel_per_site: Optional[int], seed: int) -> pd.DataFrame:
    pool = subset_excluding_keys(df, exclude_keys)
    pool = pool[pool["label"] != EVENT_LABEL].copy()
    if max_unlabel_per_site is not None and max_unlabel_per_site > 0:
        labeled = pool[pool["label"] != UNLABEL_LABEL].copy()
        unlabel = pool[pool["label"] == UNLABEL_LABEL].copy()
        kept = []
        rng = np.random.default_rng(seed)
        for site, g in unlabel.groupby("site", sort=False):
            if len(g) <= max_unlabel_per_site:
                kept.append(g)
            else:
                kept.append(g.iloc[rng.choice(len(g), size=max_unlabel_per_site, replace=False)])
        pool = pd.concat([labeled] + kept, ignore_index=True) if kept else labeled
    return pool.reset_index(drop=True)


def save_bundle(out_dir: Path, pretrain: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, summary: Dict) -> None:
    ensure_dir(out_dir)
    for name, df in [("pretrain", pretrain), ("train", train), ("val", val), ("test", test)]:
        df.drop(columns=["split_key"], errors="ignore").to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def min_counts_stage1(args, site: str) -> Dict[str, Dict[int, int]]:
    val_event = args.min_val_event_utah2019 if site == "utah_2019" and args.min_val_event_utah2019 is not None else args.min_val_event
    test_event = args.min_test_event_utah2019 if site == "utah_2019" and args.min_test_event_utah2019 is not None else args.min_test_event
    return {
        "train": {NOISE_LABEL: args.min_train_noise, EVENT_LABEL: args.min_train_event},
        "val": {NOISE_LABEL: args.min_val_noise, EVENT_LABEL: val_event},
        "test": {NOISE_LABEL: args.min_test_noise, EVENT_LABEL: test_event},
    }


def combine_site_splits(site_splits: Dict[str, Dict[str, Set[str]]], sites: Iterable[str], split: str) -> Set[str]:
    out: Set[str] = set()
    for site in sites:
        out.update(site_splits[site][split])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Create ratio-balanced group-aware 3-site experiment splits.")
    ap.add_argument("--all_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sites", nargs="+", default=["pohang", "utah_2019", "utah_2023"])
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_split_tries", type=int, default=300)
    ap.add_argument("--max_unlabel_per_site", type=int, default=3000)
    ap.add_argument("--min_train_noise", type=int, default=1)
    ap.add_argument("--min_train_event", type=int, default=1)
    ap.add_argument("--min_val_noise", type=int, default=1)
    ap.add_argument("--min_val_event", type=int, default=1)
    ap.add_argument("--min_test_noise", type=int, default=1)
    ap.add_argument("--min_test_event", type=int, default=1)
    ap.add_argument("--min_val_event_utah2019", type=int, default=20)
    ap.add_argument("--min_test_event_utah2019", type=int, default=20)
    args = ap.parse_args()

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"ratios must sum to 1.0: {ratios}")

    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    df = pd.read_csv(args.all_csv)
    validate_df(df)
    df = add_split_key(df)
    sites = [s for s in args.sites if s in set(df["site"].astype(str))]
    if not sites:
        raise RuntimeError("No requested sites present")
    max_unlabel = args.max_unlabel_per_site if args.max_unlabel_per_site > 0 else None

    site_splits: Dict[str, Dict[str, Set[str]]] = {}
    site_stats: Dict[str, Dict] = {}
    for idx, site in enumerate(sites):
        df_site = df[df["site"] == site].copy()
        split_keys, used_seed, stats = make_site_split(
            df_site=df_site,
            ratios=ratios,
            seed=args.seed + idx * 10000,
            max_tries=args.max_split_tries,
            min_counts=min_counts_stage1(args, site),
        )
        site_splits[site] = split_keys
        site_stats[site] = stats

        df_site_labeled = labeled_only(df_site)
        train = subset_by_keys(df_site_labeled, split_keys["train"])
        val = subset_by_keys(df_site_labeled, split_keys["val"])
        test = subset_by_keys(df_site_labeled, split_keys["test"])
        pretrain = build_pretrain(df_site, split_keys["test"], max_unlabel, used_seed)
        save_bundle(
            out_root / f"stage1_{site}_only",
            pretrain, train, val, test,
            {
                "experiment": f"stage1_{site}_only",
                "split_policy": "ratio_balanced_group_aware_by_site",
                "ratios": ratios,
                "site": site,
                "site_split_stats": stats,
                "pretrain": summarize_df(pretrain),
                "train": summarize_df(train),
                "val": summarize_df(val),
                "test": summarize_df(test),
            },
        )

    # Stage2: union of independently balanced site splits, preserving each site's ratio.
    df_labeled_all = labeled_only(df)
    train_keys = combine_site_splits(site_splits, sites, "train")
    val_keys = combine_site_splits(site_splits, sites, "val")
    test_keys = combine_site_splits(site_splits, sites, "test")
    save_bundle(
        out_root / "stage2_joint_all",
        build_pretrain(df, test_keys, max_unlabel, args.seed),
        subset_by_keys(df_labeled_all, train_keys),
        subset_by_keys(df_labeled_all, val_keys),
        subset_by_keys(df_labeled_all, test_keys),
        {
            "experiment": "stage2_joint_all",
            "split_policy": "union_of_site_ratio_balanced_splits",
            "ratios": ratios,
            "site_split_stats": site_stats,
            "pretrain": summarize_df(build_pretrain(df, test_keys, max_unlabel, args.seed)),
            "train": summarize_df(subset_by_keys(df_labeled_all, train_keys)),
            "val": summarize_df(subset_by_keys(df_labeled_all, val_keys)),
            "test": summarize_df(subset_by_keys(df_labeled_all, test_keys)),
        },
    )

    # Stage3: source train/val from source balanced split, target test from target balanced test.
    for src in sites:
        for tgt in sites:
            if src == tgt:
                continue
            df_src = df[df["site"] == src].copy()
            train = subset_by_keys(df_labeled_all, site_splits[src]["train"])
            val = subset_by_keys(df_labeled_all, site_splits[src]["val"])
            test = subset_by_keys(df_labeled_all, site_splits[tgt]["test"])
            pretrain = build_pretrain(df_src, set(), max_unlabel, args.seed)
            save_bundle(
                out_root / f"stage3_{src}_to_{tgt}",
                pretrain, train, val, test,
                {
                    "experiment": f"stage3_{src}_to_{tgt}",
                    "split_policy": "source_train_val_from_site_balanced_split_target_test_from_target_balanced_test",
                    "source_site": src,
                    "target_site": tgt,
                    "pretrain": summarize_df(pretrain),
                    "train": summarize_df(train),
                    "val": summarize_df(val),
                    "test": summarize_df(test),
                },
            )

    # Stage4: leave one out, source train/val are unions of balanced source-site train/val, heldout test is heldout balanced test.
    for heldout in sites:
        src_sites = [s for s in sites if s != heldout]
        train = subset_by_keys(df_labeled_all, combine_site_splits(site_splits, src_sites, "train"))
        val = subset_by_keys(df_labeled_all, combine_site_splits(site_splits, src_sites, "val"))
        test = subset_by_keys(df_labeled_all, site_splits[heldout]["test"])
        pretrain = build_pretrain(df[df["site"].isin(src_sites)].copy(), set(), max_unlabel, args.seed)
        save_bundle(
            out_root / f"stage4_leave_one_site_out_{heldout}",
            pretrain, train, val, test,
            {
                "experiment": f"stage4_leave_one_site_out_{heldout}",
                "split_policy": "source_union_of_site_balanced_train_val_heldout_balanced_test",
                "heldout_site": heldout,
                "source_sites": src_sites,
                "pretrain": summarize_df(pretrain),
                "train": summarize_df(train),
                "val": summarize_df(val),
                "test": summarize_df(test),
            },
        )

    payload = {
        "split_policy": "ratio_balanced_group_aware_by_site",
        "all_csv": str(Path(args.all_csv).resolve()),
        "out_dir": str(out_root.resolve()),
        "sites": sites,
        "ratios": ratios,
        "seed": int(args.seed),
        "max_split_tries": int(args.max_split_tries),
        "max_unlabel_per_site": max_unlabel,
        "site_split_stats": site_stats,
    }
    (out_root / "split_policy_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved ratio-balanced splits under: {out_root}")


if __name__ == "__main__":
    main()
