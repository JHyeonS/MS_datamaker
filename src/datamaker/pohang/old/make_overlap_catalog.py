import pandas as pd

input_csv = "outputs/event_overlap_check.csv"
output_xlsx = "outputs/overlap_catalog.xlsx"

df = pd.read_csv(input_csv)
dt = pd.to_datetime(df["recovered_time_utc"])

catalog = pd.DataFrame()
catalog["UTC"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]

catalog.to_excel(output_xlsx, index=False, header=True)

print("saved:", output_xlsx)
print(catalog.head())