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


DEFAULT_BANDS = [
    (0.0, 1.0),
    (1.0, 3.0),
    (3.0, 10.0),
    (10.0, 50.0),
    (50.0, 100.0),
    (100.0, 250.0),
    (250.0, None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detail figures for raw frequency energy analysis.")
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_freq", type=float, default=None)
    return parser.parse_args()


def load_site_label(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "raw_frequency_energy_by_site_label.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "site_label" not in df.columns:
        raise ValueError(f"Missing site_label column in {path}")
    return df


def plot_log_no_dc(df: pd.DataFrame, out_path: Path, max_freq: float | None) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    for site_label, g in df.groupby("site_label"):
        g = g[g["freq_hz"] > 0].copy()
        if max_freq is not None:
            g = g[g["freq_hz"] <= max_freq]
        g = g.sort_values("freq_hz")
        ax.plot(g["freq_hz"], g["power_fraction"], linewidth=1.25, label=str(site_label))

    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz), DC excluded")
    ax.set_ylabel("Power fraction per FFT bin (log)")
    ax.set_title("Raw Frequency Energy by Site/Label, Log Scale without DC")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def band_label(lo: float, hi: float | None) -> str:
    if hi is None:
        return f"{lo:g}+"
    return f"{lo:g}-{hi:g}"


def make_band_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site_label, g in df.groupby("site_label"):
        max_freq = float(g["freq_hz"].max())
        for lo, hi in DEFAULT_BANDS:
            upper = max_freq if hi is None else hi
            if hi is None:
                sub = g[g["freq_hz"] >= lo]
            else:
                sub = g[(g["freq_hz"] >= lo) & (g["freq_hz"] < hi)]
            rows.append(
                {
                    "site_label": site_label,
                    "band_hz": band_label(lo, hi),
                    "low_hz": lo,
                    "high_hz": upper,
                    "power_fraction_sum": float(sub["power_fraction"].sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_band_bars(bands: pd.DataFrame, out_path: Path) -> None:
    pivot = bands.pivot(index="band_hz", columns="site_label", values="power_fraction_sum")
    pivot = pivot.loc[[band_label(lo, hi) for lo, hi in DEFAULT_BANDS if band_label(lo, hi) in pivot.index]]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_xlabel("Frequency band (Hz)")
    ax.set_ylabel("Summed power fraction")
    ax.set_title("Raw Frequency Energy by Band")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cumulative(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    for site_label, g in df.groupby("site_label"):
        g = g.sort_values("freq_hz").copy()
        g["cum_power_fraction"] = g["power_fraction"].cumsum()
        ax.plot(g["freq_hz"], g["cum_power_fraction"], linewidth=1.4, label=str(site_label))

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cumulative power fraction")
    ax.set_title("Raw Frequency Cumulative Energy")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "figures_detail"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_site_label(analysis_dir)
    bands = make_band_frame(df)
    bands.to_csv(out_dir / "raw_frequency_band_summary.csv", index=False)

    plot_log_no_dc(df, out_dir / "raw_frequency_log_no_dc.png", args.max_freq)
    plot_band_bars(bands, out_dir / "raw_frequency_band_summary.png")
    plot_cumulative(df, out_dir / "raw_frequency_cumulative.png")

    index = [
        out_dir / "raw_frequency_log_no_dc.png",
        out_dir / "raw_frequency_band_summary.png",
        out_dir / "raw_frequency_cumulative.png",
        out_dir / "raw_frequency_band_summary.csv",
    ]
    (out_dir / "raw_frequency_detail_figures.txt").write_text(
        "\n".join(str(p) for p in index) + "\n",
        encoding="utf-8",
    )
    print("\n".join(str(p) for p in index))


if __name__ == "__main__":
    main()
