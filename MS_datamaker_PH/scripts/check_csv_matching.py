# check_csv_matching.py
import pandas as pd

excel_path = r"C:\Users\ted01\OneDrive\Desktop\학부\학부때 연구관련\deep-sad-eq-detection\hong\image\iDAS_detection_korean_microeqrthquake_list_20190501_20191001.xlsx"
rec_path = r"C:\Users\ted01\OneDrive\Desktop\recovered_catalog_debug.csv"
out_path = r"C:\Users\ted01\OneDrive\Desktop\event_overlap_check.csv"

TOL = 60  # 초

# Excel: 실제 헤더는 2번째 줄
df_excel = pd.read_excel(excel_path, header=1)
df_excel = df_excel[df_excel["UTC"].notna()].copy()
df_excel["datetime"] = pd.to_datetime(df_excel["UTC"])

# recovered: UTC 기준 비교
df_rec = pd.read_csv(rec_path)
df_rec["datetime"] = pd.to_datetime(df_rec["start_datetime_utc"])

matches = []
for _, r in df_rec.iterrows():
    dt = r["datetime"]
    diff = (df_excel["datetime"] - dt).abs().dt.total_seconds()
    hit = df_excel[diff <= TOL]
    if len(hit) > 0:
        j = diff.idxmin()
        matches.append({
            "recovered_time_utc": dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "excel_time": df_excel.loc[j, "datetime"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "time_diff_sec": float(diff.loc[j]),
            "cluster_id": int(r["cluster_id"]),
            "cluster_size": int(r["cluster_size"]),
            "duration_sec": float(r["duration_sec"]),
            "repr_filename": r["repr_filename"],
        })

df_match = pd.DataFrame(matches)
df_match.to_csv(out_path, index=False, encoding="utf-8-sig")

print("matched events:", len(df_match))
print("saved:", out_path)