#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(r"C:\Users\ted01\OneDrive\Desktop\temp_for_xftp\pohang")
OUT_CSV = ROOT.parent / "pohang_labels.csv"
SEGMENT_SEC = 1.0


def parse_png_info(png_path: Path):
    parent = png_path.parent.name                    # 예: 0000_xxx
    stem = re.sub(r"^\d+_", "", parent)             # 예: xxx
    tdms_file = stem + ".tdms"

    m = re.search(r"seg_(\d+)\.png$", png_path.name)
    if m is None:
        raise ValueError(f"Unexpected png name: {png_path.name}")

    seg_idx = int(m.group(1))
    start_sec = seg_idx * SEGMENT_SEC
    end_sec = (seg_idx + 1) * SEGMENT_SEC

    return {
        "site": "pohang",
        "tdms_file": tdms_file,
        "seg_png": str(png_path.relative_to(ROOT)),
        "segment_index": seg_idx,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def load_existing_labels(csv_path: Path):
    labeled = {}
    if not csv_path.exists():
        return labeled

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labeled[row["seg_png"]] = row
    return labeled


def save_all_rows(csv_path: Path, rows_dict):
    fieldnames = [
        "site", "tdms_file", "seg_png", "segment_index",
        "start_sec", "end_sec", "is_event", "confidence", "note"
    ]
    rows = sorted(rows_dict.values(), key=lambda x: x["seg_png"])

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def show_image_and_label(png_path: Path, existing_row=None):
    img = mpimg.imread(png_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    title = f"{png_path.parent.name}/{png_path.name}"
    if existing_row is not None:
        title += f"\n[EXISTING] is_event={existing_row['is_event']}, conf={existing_row['confidence']}, note={existing_row['note']}"
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.show(block=False)

    print("\n" + "=" * 80)
    print(f"파일: {png_path}")
    print("입력: 1=event, 0=noise, u=uncertain, s=skip, b=back, q=quit")
    key = input("label> ").strip().lower()

    plt.close()

    return key


def main():
    png_list = sorted(ROOT.rglob("seg_*.png"))
    if len(png_list) == 0:
        print(f"[ERROR] No PNG files found under: {ROOT}")
        return

    rows_dict = load_existing_labels(OUT_CSV)
    print(f"[INFO] total png: {len(png_list)}")
    print(f"[INFO] existing labels: {len(rows_dict)}")
    print(f"[INFO] csv path: {OUT_CSV}")

    idx = 0
    while idx < len(png_list):
        png_path = png_list[idx]
        info = parse_png_info(png_path)
        existing_row = rows_dict.get(info["seg_png"], None)

        key = show_image_and_label(png_path, existing_row=existing_row)

        if key == "q":
            print("[INFO] Quit and save.")
            break
        elif key == "b":
            idx = max(0, idx - 1)
            continue
        elif key == "s":
            idx += 1
            continue
        elif key not in ["1", "0", "u"]:
            print("[WARN] invalid input. use 1 / 0 / u / s / b / q")
            continue

        if key == "1":
            is_event = 1
        elif key == "0":
            is_event = 0
        else:
            is_event = -1

        confidence = input("confidence (1~3, empty=2)> ").strip()
        if confidence == "":
            confidence = "2"

        note = input("note (empty ok)> ").strip()

        row = {
            "site": info["site"],
            "tdms_file": info["tdms_file"],
            "seg_png": info["seg_png"],
            "segment_index": info["segment_index"],
            "start_sec": info["start_sec"],
            "end_sec": info["end_sec"],
            "is_event": is_event,
            "confidence": confidence,
            "note": note,
        }

        rows_dict[info["seg_png"]] = row
        save_all_rows(OUT_CSV, rows_dict)

        print(f"[SAVE] {info['seg_png']} -> is_event={is_event}, conf={confidence}")
        idx += 1

    save_all_rows(OUT_CSV, rows_dict)
    print(f"[DONE] saved csv: {OUT_CSV}")


if __name__ == "__main__":
    main()