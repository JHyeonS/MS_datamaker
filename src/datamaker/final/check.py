#!/usr/bin/env python3
import pandas as pd

csv_path = "/home/ted1204/MS_Detection_Datamaker/MS_datamaker_FINAL/output_npy/metadata/all_samples.csv"
site_name = "utah_2019"

df = pd.read_csv(csv_path)
df = df[df["site"] == site_name].copy()
df["group_id"] = df["group_id"].astype(str)

g = (
    df[df["label"].isin([0, 1])]
    .groupby(["group_id", "label"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

if 0 not in g.columns:
    g[0] = 0
if 1 not in g.columns:
    g[1] = 0

g = g.rename(columns={0: "noise", 1: "event"})
g["total"] = g["noise"] + g["event"]
g = g.sort_values(["event", "total"], ascending=[False, False]).reset_index(drop=True)

print(g.head(50))
print()
print("num event groups:", int((g["event"] > 0).sum()))
print("total event samples:", int(g["event"].sum()))
print("top 10 event groups sum:", int(g["event"].head(10).sum()))