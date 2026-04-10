#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

csv_path = "/home/ted1204/MS_datamaker_merge/outputs_npy/metadata/all_samples.csv"

df = pd.read_csv(csv_path)
df = df[df["site"].astype(str).str.lower() == "utah"].copy()

# split_key 재구성
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
df["split_key"] = df["site"].astype(str) + "::" + pd.Series(split_key, index=df.index).astype(str)

# labeled만
df_labeled = df[df["label"].isin([0, 1])].copy()
df_event = df[df["label"] == 1].copy()

print("=== BASIC COUNTS ===")
print("UTAH total rows:", len(df))
print("UTAH labeled rows:", len(df_labeled))
print("UTAH event rows:", len(df_event))
print("UTAH total groups:", df["split_key"].nunique())
print("UTAH labeled groups:", df_labeled["split_key"].nunique())
print("UTAH event-containing groups:", df_event["split_key"].nunique())
print()

print("=== EVENT GROUP COUNTS ===")
g_event = (
    df_event.groupby("split_key")
    .size()
    .reset_index(name="n_event")
    .sort_values("n_event", ascending=False)
)
print(g_event.to_string(index=False))
print()

print("=== LABELED GROUP TABLE (noise/event counts) ===")
g_all = (
    df_labeled.groupby(["split_key", "label"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

if 0 not in g_all.columns:
    g_all[0] = 0
if 1 not in g_all.columns:
    g_all[1] = 0

g_all = g_all.rename(columns={0: "n_noise", 1: "n_event"})
g_all = g_all.sort_values(["n_event", "n_noise"], ascending=[False, False])
print(g_all.to_string(index=False))