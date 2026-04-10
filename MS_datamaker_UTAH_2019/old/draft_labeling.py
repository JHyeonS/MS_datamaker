import csv
from pathlib import Path
import re

root = Path(r"C:\Users\ted01\OneDrive\Desktop\temp_for_xftp\pohang")
out_csv = root.parent / "pohang_labels_template.csv"
segment_sec = 1.0

rows = []
for png in sorted(root.rglob("seg_*.png")):
    parent = png.parent.name                      # 0000_xxx
    stem = re.sub(r"^\d+_", "", parent)          # xxx
    tdms_file = stem + ".tdms"

    m = re.search(r"seg_(\d+)\.png$", png.name)
    seg_idx = int(m.group(1))
    start_sec = seg_idx * segment_sec
    end_sec = (seg_idx + 1) * segment_sec

    rows.append([
        "pohang",
        tdms_file,
        str(png.relative_to(root)),
        seg_idx,
        start_sec,
        end_sec,
        "",   # is_event
        "",   # confidence
        ""    # note
    ])

with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "site", "tdms_file", "seg_png", "segment_index",
        "start_sec", "end_sec", "is_event", "confidence", "note"
    ])
    writer.writerows(rows)

print(f"saved: {out_csv}")