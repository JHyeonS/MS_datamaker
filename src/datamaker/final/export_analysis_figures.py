#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def short_run_name(run_dir: Path) -> str:
    name = run_dir.name
    if name.endswith("_3site"):
        name = name[:-6]
    return name


def plot_frequency_by_scope(metadata_dir: Path, out_dir: Path, run_label: str) -> Path:
    df = pd.read_csv(metadata_dir / "frequency_energy_by_scope.csv")
    out_path = out_dir / "frequency_energy_by_scope.png"

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for scope, g in df.groupby("scope"):
        g = g.sort_values("freq_hz")
        ax.plot(g["freq_hz"], g["power_fraction"], linewidth=1.7, label=str(scope))

    ax.set_title(f"{run_label} | Frequency Energy by Site", fontsize=11)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power fraction")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_frequency_by_site_label(metadata_dir: Path, out_dir: Path, run_label: str) -> Path:
    df = pd.read_csv(metadata_dir / "frequency_energy_by_site_label.csv")
    out_path = out_dir / "frequency_energy_by_site_label.png"

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), dpi=150, sharex=True)
    label_order = ["event", "noise", "unlabel"]

    for ax, label_name in zip(axes, label_order):
        sub = df[df["site_label"].astype(str).str.endswith("|" + label_name)]
        for site_label, g in sub.groupby("site_label"):
            site = str(site_label).split("|", 1)[0]
            g = g.sort_values("freq_hz")
            ax.plot(g["freq_hz"], g["power_fraction"], linewidth=1.4, label=site)
        ax.set_title(label_name, fontsize=10)
        ax.set_ylabel("Power fraction")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=3, fontsize=8)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"{run_label} | Frequency Energy by Site and Label", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def load_final_stats(metadata_dir: Path) -> dict:
    with (metadata_dir / "preprocess_final_stats.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_min_max_by_site(metadata_dir: Path, out_dir: Path, run_label: str) -> Path:
    stats = load_final_stats(metadata_dir)
    by_scope = stats.get("final_stats_by_scope") or stats.get("by_scope") or {}
    rows = []
    for site, values in by_scope.items():
        rows.append(
            {
                "site": site,
                "min": float(values["min"]),
                "max": float(values["max"]),
                "rms": float(values["rms"]),
            }
        )
    df = pd.DataFrame(rows).sort_values("site")
    out_path = out_dir / "rms_min_max_by_site.png"

    x = np.arange(len(df))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    ax1.bar(x - width / 2, df["min"], width, label="min", color="#4C78A8")
    ax1.bar(x + width / 2, df["max"], width, label="max", color="#F58518")
    ax1.axhline(7, color="#D62728", linestyle="--", linewidth=1, alpha=0.8)
    ax1.axhline(-7, color="#D62728", linestyle="--", linewidth=1, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["site"], rotation=0)
    ax1.set_ylabel("Value after RMS normalization")
    ax1.set_title(f"{run_label} | Min/Max after RMS Normalization", fontsize=11)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(x, df["rms"], color="#54A24B", marker="o", linewidth=1.5, label="RMS")
    ax2.set_ylabel("RMS")
    ax2.set_ylim(0, max(1.2, float(df["rms"].max()) * 1.2))
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_exceedance_by_site_label(metadata_dir: Path, out_dir: Path, run_label: str) -> Path:
    df = pd.read_csv(metadata_dir / "sample_value_exceedance_summary_by_site_label_abs_gt_7p0.csv")
    df = df.sort_values(["site", "label_name"])
    out_path = out_dir / "abs_gt_7_exceedance_by_site_label.png"

    labels = df["site"].astype(str) + "|" + df["label_name"].astype(str)
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=150)
    bars = ax1.bar(x, df["exceed_samples"], color="#E45756", alpha=0.85, label="exceed count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("Samples with abs(min/max) > 7")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.set_title(f"{run_label} | abs(min/max) > 7 by Site and Label", fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(x, df["exceed_fraction"], color="#2F4B7C", marker="o", linewidth=1.5, label="fraction")
    ax2.set_ylabel("Fraction")
    ax2.set_ylim(0, 1)

    for bar, frac in zip(bars, df["exceed_fraction"]):
        if bar.get_height() > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{frac:.1%}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax1.legend(frameon=False, loc="upper left")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def export_figures(run_dir: Path, png_root: Path) -> list[Path]:
    metadata_dir = run_dir / "metadata"
    out_dir = png_root / run_dir.name / "analysis"
    ensure_dir(out_dir)
    run_label = short_run_name(run_dir)

    outputs = [
        plot_frequency_by_scope(metadata_dir, out_dir, run_label),
        plot_frequency_by_site_label(metadata_dir, out_dir, run_label),
        plot_min_max_by_site(metadata_dir, out_dir, run_label),
        plot_exceedance_by_site_label(metadata_dir, out_dir, run_label),
    ]

    index_path = out_dir / "analysis_figures.txt"
    index_path.write_text("\n".join(str(p) for p in outputs) + "\n", encoding="utf-8")
    outputs.append(index_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export analysis metadata figures.")
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--png_root", default="output_npy/final/png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run_dir_str in args.run_dir:
        outputs = export_figures(Path(run_dir_str), Path(args.png_root))
        print(f"[DONE] {run_dir_str}")
        for path in outputs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
