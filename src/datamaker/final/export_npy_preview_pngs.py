#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def select_samples(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()

    rng = np.random.default_rng(seed)
    groups = []
    for _, g in df.groupby(["site", "label", "label_name"], dropna=False):
        shuffled = g.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        groups.append(shuffled.reset_index(drop=True))

    selected_rows = []
    cursor = 0
    while len(selected_rows) < n:
        progressed = False
        for g in groups:
            if cursor < len(g):
                selected_rows.append(g.iloc[cursor])
                progressed = True
                if len(selected_rows) >= n:
                    break
        if not progressed:
            break
        cursor += 1

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def robust_limits(arr: np.ndarray, percentile: float) -> tuple[float, float]:
    lo = float(np.nanpercentile(arr, 100.0 - percentile))
    hi = float(np.nanpercentile(arr, percentile))
    scale = max(abs(lo), abs(hi), 1e-12)
    return -scale, scale


def save_preview(row: pd.Series, out_root: Path, percentile: float, dpi: int) -> Path:
    npy_path = Path(str(row["npy_path"]))
    arr = np.load(npy_path)
    arr = np.asarray(arr, dtype=np.float32)

    site = str(row["site"])
    label = int(row["label"])
    label_name = str(row["label_name"])
    label_dir = f"{label}_{label_name}"
    out_dir = out_root / site / label_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{npy_path.stem}.png"
    vmin, vmax = robust_limits(arr, percentile)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    im = ax.imshow(arr, aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(
        f"{site} / {label_dir} / {npy_path.stem} | shape={arr.shape}",
        fontsize=9,
    )
    ax.set_xlabel("sample")
    ax.set_ylabel("channel")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    return png_path


def export_pngs(all_csv: Path, out_root: Path, n: int, seed: int, percentile: float, dpi: int) -> Path:
    df = pd.read_csv(all_csv)
    required = {"npy_path", "site", "label", "label_name"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {all_csv}: {missing}")

    selected = select_samples(df, n=n, seed=seed)
    rows = []
    for i, row in selected.iterrows():
        png_path = save_preview(row, out_root=out_root, percentile=percentile, dpi=dpi)
        rows.append(
            {
                "preview_index": int(i),
                "site": str(row["site"]),
                "label": int(row["label"]),
                "label_name": str(row["label_name"]),
                "group_id": str(row["group_id"]) if "group_id" in row else "",
                "npy_path": str(row["npy_path"]),
                "png_path": str(png_path),
            }
        )

    index_path = out_root / "preview_index.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export preview PNGs from final NPY dataset.")
    parser.add_argument("--all_csv", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=130)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = export_pngs(
        all_csv=Path(args.all_csv),
        out_root=Path(args.out_root),
        n=int(args.n),
        seed=int(args.seed),
        percentile=float(args.percentile),
        dpi=int(args.dpi),
    )
    print(f"[DONE] wrote preview index: {index_path}")


if __name__ == "__main__":
    main()
