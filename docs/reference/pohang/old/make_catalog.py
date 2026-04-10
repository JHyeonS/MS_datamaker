# make_catalog.py
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

IMG_DIRS = [
    r"C:\Users\ted01\OneDrive\Desktop\학부\학부때 연구관련\deep-sad-eq-detection\hong\image\deep_svdd_test\1_event",
    r"C:\Users\ted01\OneDrive\Desktop\학부\학부때 연구관련\deep-sad-eq-detection\hong\image\das_image_old\1_event",
    r"C:\Users\ted01\OneDrive\Desktop\학부\학부때 연구관련\deep-sad-eq-detection\hong\image\das_image\1_event",
    r"C:\Users\ted01\OneDrive\Desktop\학부\학부때 연구관련\deep-sad-eq-detection\hong\image\1026_train\event",
    r"C:\Users\ted01\OneDrive\Desktop\학부\학부때 연구관련\deep-sad-eq-detection\hong\image\1026_test\1_event",
    r"C:\Users\ted01\OneDrive\Desktop\학부\학부때 연구관련\deep-sad-eq-detection\hong\image\1010_test\event",
]

OUT_CSV = r"C:\Users\ted01\OneDrive\Desktop\recovered_catalog.csv"
OUT_DEBUG_CSV = r"C:\Users\ted01\OneDrive\Desktop\recovered_catalog_debug.csv"

GAP_SEC = 3.0           # cluster 기준
KST_TO_UTC_HOURS = 9    # 이미지명이 KST라고 확인됨

def parse_png_time(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    try:
        # 예: 190917_184042_000
        return datetime.strptime(stem, "%y%m%d_%H%M%S_%f")
    except ValueError:
        return None

rows = []
for img_dir in IMG_DIRS:
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    for f in files:
        dt_kst = parse_png_time(f)
        if dt_kst is None:
            continue
        rows.append({
            "filepath": f,
            "filename": os.path.basename(f),
            "datetime_kst": dt_kst,
            "source_dir": img_dir,
        })

if len(rows) == 0:
    raise RuntimeError("유효한 PNG timestamp를 찾지 못했습니다.")

df = pd.DataFrame(rows).sort_values("datetime_kst").reset_index(drop=True)

# 같은 시각 중복 제거
df_unique = (
    df.groupby("datetime_kst", as_index=False)
      .agg(
          filename=("filename", "first"),
          filepath=("filepath", "first"),
          source_count=("source_dir", "nunique"),
          source_dirs=("source_dir", lambda x: " | ".join(sorted(set(x))))
      )
      .sort_values("datetime_kst")
      .reset_index(drop=True)
)

# cluster 복원
clusters = []
cur = [df_unique.iloc[0]]

for i in range(1, len(df_unique)):
    prev_dt = cur[-1]["datetime_kst"]
    now_row = df_unique.iloc[i]
    now_dt = now_row["datetime_kst"]

    if (now_dt - prev_dt).total_seconds() <= GAP_SEC:
        cur.append(now_row)
    else:
        clusters.append(cur)
        cur = [now_row]

clusters.append(cur)

catalog_rows = []
debug_rows = []

for cid, cluster in enumerate(clusters):
    start_kst = cluster[0]["datetime_kst"]
    end_kst = cluster[-1]["datetime_kst"]

    # workflow 호환 위해 UTC로 변환해서 catalog 저장
    start_utc = start_kst
    end_utc = end_kst 

    catalog_rows.append({
        "date": start_utc.strftime("%y%m%d"),
        "time": start_utc.strftime("%H%M%S"),
        "ms": start_utc.strftime("%f")[:3],
    })

    debug_rows.append({
        "cluster_id": cid,
        "start_datetime_kst": start_kst.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "end_datetime_kst": end_kst.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "start_datetime_utc": start_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "end_datetime_utc": end_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "cluster_size": len(cluster),
        "duration_sec": round((end_kst - start_kst).total_seconds(), 3),
        "source_count_sum": int(sum(r["source_count"] for r in cluster)),
        "repr_filename": cluster[0]["filename"],
    })

catalog_df = pd.DataFrame(catalog_rows).sort_values(["date", "time", "ms"]).reset_index(drop=True)
debug_df = pd.DataFrame(debug_rows)

catalog_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
debug_df.to_csv(OUT_DEBUG_CSV, index=False, encoding="utf-8-sig")

print(f"saved catalog: {OUT_CSV} ({len(catalog_df)} events)")
print(f"saved debug  : {OUT_DEBUG_CSV} ({len(debug_df)} clusters)")