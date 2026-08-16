#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_NAMES = {"0": "noise", "1": "event", "2": "unlabel"}


def load_manifest(metadata_csv: Path, repo_root: Path):
    rows = []
    with open(metadata_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            site = row["site"]
            label = str(row["label"])
            p = Path(row["npy_path"])
            if not p.exists():
                alt = repo_root / "output_npy" / site / f"{label}_{LABEL_NAMES.get(label, row.get('label_name', label))}" / p.name
                p = alt
            row["resolved_npy_path"] = str(p)
            rows.append(row)
    return rows


def choose_rows(rows, sites, labels, max_per_label, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        if row["site"] in sites and str(row["label"]) in labels:
            buckets[(row["site"], str(row["label"]))].append(row)

    chosen = []
    for key in sorted(buckets):
        vals = buckets[key]
        rng.shuffle(vals)
        chosen.extend(vals[:max_per_label] if max_per_label > 0 else vals)
    return chosen


def channel_mean_psd(arr, fs, nperseg):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array (channels,time), got shape={arr.shape}")
    arr = arr - np.mean(arr, axis=1, keepdims=True)
    nperseg = min(int(nperseg), arr.shape[1])
    freqs, psd = welch(
        arr,
        fs=float(fs),
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=1,
        detrend=False,
        scaling="density",
    )
    return freqs, np.mean(psd, axis=0)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def plot_mean_psd(out_dir, mean_psd):
    for site in sorted({k[0] for k in mean_psd}):
        plt.figure(figsize=(10, 6))
        for label in ["0", "1", "2"]:
            key = (site, label)
            if key not in mean_psd:
                continue
            freqs, vals, n = mean_psd[key]
            plt.semilogy(freqs[1:], vals[1:], label=f"{LABEL_NAMES.get(label, label)} (n={n})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mean PSD")
        plt.title(f"Mean temporal PSD - {site}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{site}_mean_psd.png", dpi=160)
        plt.close()

    plt.figure(figsize=(11, 7))
    for key in sorted(mean_psd):
        site, label = key
        if label not in ("0", "1"):
            continue
        freqs, vals, n = mean_psd[key]
        plt.semilogy(freqs[1:], vals[1:], label=f"{site}-{LABEL_NAMES.get(label, label)}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean PSD")
    plt.title("Mean temporal PSD by site and class")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "all_sites_noise_event_mean_psd.png", dpi=170)
    plt.close()


def plot_examples(out_dir, example_rows, fs_default, nperseg, examples_per_label):
    rng = random.Random(1234)
    buckets = defaultdict(list)
    for row in example_rows:
        if str(row["label"]) in ("0", "1"):
            buckets[(row["site"], str(row["label"]))].append(row)

    for key in sorted(buckets):
        site, label = key
        vals = buckets[key]
        rng.shuffle(vals)
        vals = vals[:examples_per_label]

        plt.figure(figsize=(10, 6))
        for row in vals:
            p = Path(row["resolved_npy_path"])
            if not p.exists():
                continue
            arr = np.load(p)
            fs = float(row.get("saved_fs") or fs_default)
            freqs, psd = channel_mean_psd(arr, fs=fs, nperseg=nperseg)
            plt.semilogy(freqs[1:], psd[1:], alpha=0.85, label=p.stem)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Channel-mean PSD")
        plt.title(f"Example spectra - {site} {LABEL_NAMES.get(label, label)}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"example_spectra_{site}_{LABEL_NAMES.get(label, label)}.png", dpi=160)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default="/home/ted1204/MS_datamaker/MS_datamaker_FINAL")
    ap.add_argument("--metadata_csv", default="output_npy/metadata/all_samples.csv")
    ap.add_argument("--out_dir", default="analysis/psd_by_site")
    ap.add_argument("--sites", nargs="+", default=["pohang", "utah_2019", "utah_2023"])
    ap.add_argument("--labels", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--max_per_label", type=int, default=500)
    ap.add_argument("--examples_per_label", type=int, default=6)
    ap.add_argument("--nperseg", type=int, default=512)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.is_absolute():
        metadata_csv = repo_root / metadata_csv
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(metadata_csv, repo_root=repo_root)
    chosen = choose_rows(rows, args.sites, [str(x) for x in args.labels], args.max_per_label, args.seed)

    accum = {}
    counts = defaultdict(int)
    per_file = []
    failed = []

    for i, row in enumerate(chosen, 1):
        p = Path(row["resolved_npy_path"])
        site = row["site"]
        label = str(row["label"])
        try:
            arr = np.load(p)
            fs = float(row.get("saved_fs") or 1000.0)
            freqs, psd = channel_mean_psd(arr, fs=fs, nperseg=args.nperseg)
            key = (site, label)
            if key not in accum:
                accum[key] = np.zeros_like(psd, dtype=np.float64)
            accum[key] += psd
            counts[key] += 1

            peak_freq = float(freqs[int(np.argmax(psd))])
            centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-30))
            per_file.append({
                "site": site,
                "label": label,
                "label_name": LABEL_NAMES.get(label, row.get("label_name", label)),
                "npy_path": str(p),
                "saved_fs": fs,
                "peak_freq_hz": peak_freq,
                "centroid_hz": centroid,
                "mean_power": float(np.mean(psd)),
            })
        except Exception as e:
            failed.append({"site": site, "label": label, "npy_path": str(p), "reason": str(e)})

        if i == 1 or i % 500 == 0:
            print(f"[INFO] processed {i}/{len(chosen)}")

    mean_psd = {}
    psd_rows = []
    summary_rows = []
    for key in sorted(accum):
        site, label = key
        n = counts[key]
        vals = accum[key] / max(n, 1)
        mean_psd[key] = (freqs, vals, n)
        summary_rows.append({
            "site": site,
            "label": label,
            "label_name": LABEL_NAMES.get(label, label),
            "n_analyzed": n,
            "peak_freq_hz": float(freqs[int(np.argmax(vals))]),
            "centroid_hz": float(np.sum(freqs * vals) / (np.sum(vals) + 1e-30)),
            "mean_power": float(np.mean(vals)),
        })
        for f, v in zip(freqs, vals):
            psd_rows.append({
                "site": site,
                "label": label,
                "label_name": LABEL_NAMES.get(label, label),
                "frequency_hz": float(f),
                "mean_psd": float(v),
            })

    write_csv(out_dir / "psd_summary_by_site_label.csv", summary_rows,
              ["site", "label", "label_name", "n_analyzed", "peak_freq_hz", "centroid_hz", "mean_power"])
    write_csv(out_dir / "mean_psd_by_site_label.csv", psd_rows,
              ["site", "label", "label_name", "frequency_hz", "mean_psd"])
    write_csv(out_dir / "per_file_psd_metrics.csv", per_file,
              ["site", "label", "label_name", "npy_path", "saved_fs", "peak_freq_hz", "centroid_hz", "mean_power"])
    write_csv(out_dir / "failed_files.csv", failed, ["site", "label", "npy_path", "reason"])

    plot_mean_psd(out_dir, mean_psd)
    plot_examples(out_dir, rows, fs_default=1000.0, nperseg=args.nperseg, examples_per_label=args.examples_per_label)

    with open(out_dir / "run_info.txt", "w", encoding="utf-8") as f:
        f.write(f"metadata_csv: {metadata_csv}\n")
        f.write(f"sites: {args.sites}\n")
        f.write(f"labels: {args.labels}\n")
        f.write(f"max_per_label: {args.max_per_label}\n")
        f.write(f"examples_per_label: {args.examples_per_label}\n")
        f.write(f"nperseg: {args.nperseg}\n")
        f.write(f"chosen: {len(chosen)}\n")
        f.write(f"succeeded: {len(per_file)}\n")
        f.write(f"failed: {len(failed)}\n")

    print("Done.")
    print(f"Output: {out_dir}")
    print(f"Succeeded={len(per_file)} Failed={len(failed)}")


if __name__ == "__main__":
    main()
