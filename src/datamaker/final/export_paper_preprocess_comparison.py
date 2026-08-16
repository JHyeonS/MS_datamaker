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


SITES = ["pohang", "utah_2019", "utah_2023"]
LABELS = [(0, "noise"), (1, "event")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-candidate raw/filter-rms/filter-logenv-rms comparison figures."
    )
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--filter-rms-csv", required=True)
    parser.add_argument("--filter-logenv-rms-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples-per-site-label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--percentile", type=float, default=99.0)
    return parser.parse_args()


def normalize_site(value: str) -> str:
    value = str(value)
    return {"utah2019": "utah_2019", "utah2023": "utah_2023"}.get(value, value)


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["site"] = out["site"].map(normalize_site)
    out["label"] = out["label"].astype(int)
    out["raw_file_path"] = out["raw_file_path"].astype(str)
    for col in ["start_sec", "end_sec"]:
        out[col] = out[col].astype(float).round(6)
    for col in ["ch_start", "ch_end"]:
        out[col] = out[col].astype(int)
    return out


def key_columns() -> list[str]:
    return ["site", "label", "raw_file_path", "start_sec", "end_sec", "ch_start", "ch_end"]


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def robust_symmetric_limits(arr: np.ndarray, percentile: float) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    scale = float(np.nanpercentile(np.abs(finite), percentile))
    scale = max(scale, 1e-8)
    return -scale, scale


def select_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected = []
    for site in SITES:
        for label, label_name in LABELS:
            subset = df[(df["site"] == site) & (df["label"] == label)].copy()
            if subset.empty:
                raise ValueError(f"No matched rows for site={site} label={label_name}")
            subset = subset.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).head(n)
            selected.append(subset)
    return pd.concat(selected, ignore_index=True)


def load_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr, dtype=np.float32)


def panel_specs(row: pd.Series, raw_base: Path) -> list[tuple[str, str, Path]]:
    return [
        ("raw", "Raw", resolve_path(row["raw_npy_path"], raw_base)),
        ("filter_rms", "Filter + RMS", Path(str(row["filter_rms_npy_path"]))),
        ("filter_logenv_rms", "Filter + Log Envelope + RMS", Path(str(row["filter_logenv_rms_npy_path"]))),
    ]


def sample_metadata_title(row: pd.Series) -> str:
    return (
        f"{row['site']} / {row['label_name']} / sample {row['sample_id']} / "
        f"{float(row['start_sec']):g}-{float(row['end_sec']):g}s / ch {int(row['ch_start'])}-{int(row['ch_end'])}"
    )


def save_single_panel(
    arr: np.ndarray,
    out_path: Path,
    title: str,
    metadata_title: str,
    percentile: float,
    dpi: int,
    with_plot: bool,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = robust_symmetric_limits(arr, percentile)
    if with_plot:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=dpi, constrained_layout=True)
        im = ax.imshow(arr, cmap="seismic", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n{metadata_title}", fontsize=8)
        ax.set_xlabel("Time sample", fontsize=8)
        ax.set_ylabel("Channel", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        fig.savefig(out_path)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(5.2, 3.2), dpi=dpi, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr, cmap="seismic", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    return out_path


def save_individual_panels(row: pd.Series, out_dir: Path, raw_base: Path, percentile: float, dpi: int) -> list[Path]:
    site = str(row["site"])
    label_name = str(row["label_name"])
    sample_id = str(row["sample_id"]).zfill(4)
    metadata_title = sample_metadata_title(row)
    paths = []
    for method_key, method_title, npy_path in panel_specs(row, raw_base):
        arr = load_array(npy_path)
        base_name = f"{site}_{label_name}_{sample_id}_{method_key}.png"
        paths.append(
            save_single_panel(
                arr,
                out_dir / "individual" / "with_plot" / method_key / site / label_name / base_name,
                method_title,
                metadata_title,
                percentile,
                dpi,
                with_plot=True,
            )
        )
        paths.append(
            save_single_panel(
                arr,
                out_dir / "individual" / "clean" / method_key / site / label_name / base_name,
                method_title,
                metadata_title,
                percentile,
                dpi,
                with_plot=False,
            )
        )
    return paths


def save_triptych(row: pd.Series, out_dir: Path, raw_base: Path, percentile: float, dpi: int) -> Path:
    site = str(row["site"])
    label_name = str(row["label_name"])
    sample_id = str(row["sample_id"]).zfill(4)
    out_subdir = out_dir / site / label_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    panels = [(title, load_array(path)) for _, title, path in panel_specs(row, raw_base)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=dpi, constrained_layout=True)
    for ax, (title, arr) in zip(axes, panels):
        vmin, vmax = robust_symmetric_limits(arr, percentile)
        im = ax.imshow(arr, cmap="seismic", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time sample", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    axes[0].set_ylabel("Channel", fontsize=8)
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.suptitle(sample_metadata_title(row), fontsize=10)
    png_path = out_subdir / f"{site}_{label_name}_{sample_id}_raw_filter_rms_logenv_rms.png"
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def save_contact_sheet(index_df: pd.DataFrame, out_dir: Path, raw_base: Path, percentile: float, dpi: int) -> list[Path]:
    paths = []
    for (site, label_name), group in index_df.groupby(["site", "label_name"], sort=False):
        fig, axes = plt.subplots(len(group), 3, figsize=(10.5, 2.6 * len(group)), dpi=dpi, constrained_layout=True)
        if len(group) == 1:
            axes = np.expand_dims(axes, 0)
        for r, (_, row) in enumerate(group.iterrows()):
            panels = [(title, load_array(path)) for _, title, path in panel_specs(row, raw_base)]
            for c, (title, arr) in enumerate(panels):
                ax = axes[r, c]
                vmin, vmax = robust_symmetric_limits(arr, percentile)
                ax.imshow(arr, cmap="seismic", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
                if r == 0:
                    ax.set_title(title, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(str(row["sample_id"]).zfill(4), fontsize=8)
        fig.suptitle(f"{site} / {label_name}", fontsize=11)
        path = out_dir / f"contact_sheet_{site}_{label_name}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    raw_manifest_path = Path(args.raw_manifest)
    raw_base = raw_manifest_path.parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = normalize_key_columns(pd.read_csv(raw_manifest_path))
    filter_rms = normalize_key_columns(pd.read_csv(args.filter_rms_csv))
    logenv = normalize_key_columns(pd.read_csv(args.filter_logenv_rms_csv))

    raw = raw[raw["label"].isin([0, 1])].copy()
    raw["sample_id"] = raw["npy_path"].map(lambda p: Path(str(p)).stem)
    raw = raw.rename(columns={"npy_path": "raw_npy_path"})
    filter_rms = filter_rms.rename(columns={"npy_path": "filter_rms_npy_path"})
    logenv = logenv.rename(columns={"npy_path": "filter_logenv_rms_npy_path"})

    cols = key_columns()
    merged = raw.merge(filter_rms[cols + ["filter_rms_npy_path"]], on=cols, how="inner")
    merged = merged.merge(logenv[cols + ["filter_logenv_rms_npy_path"]], on=cols, how="inner")
    if merged.empty:
        raise ValueError("No raw/filter/logenv rows could be matched.")

    selected = select_rows(merged, n=args.samples_per_site_label, seed=args.seed)
    triptych_paths = []
    individual_paths = []
    for _, row in selected.iterrows():
        triptych_paths.append(save_triptych(row, out_dir, raw_base, args.percentile, args.dpi))
        individual_paths.extend(save_individual_panels(row, out_dir, raw_base, args.percentile, args.dpi))

    selected = selected.copy()
    selected["triptych_png_path"] = [str(p) for p in triptych_paths]
    selected.to_csv(out_dir / "paper_preprocess_comparison_index.csv", index=False)
    contact_paths = save_contact_sheet(selected, out_dir, raw_base, args.percentile, args.dpi)

    summary = {
        "matched_rows": int(len(merged)),
        "selected_rows": int(len(selected)),
        "triptych_pngs": int(len(triptych_paths)),
        "individual_pngs": int(len(individual_paths)),
        "contact_sheets": int(len(contact_paths)),
        "out_dir": str(out_dir.resolve()),
    }
    pd.Series(summary).to_json(out_dir / "paper_preprocess_comparison_summary.json", indent=2)
    print(summary)


if __name__ == "__main__":
    main()
