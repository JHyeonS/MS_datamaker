import re
import pandas as pd

path = "/home/ted1204/MS_datamaker_UTAH_2023/outputs_npy/utah_2023_2s_6split/metadata/patch_labels_v2.csv"
df = pd.read_csv(path)

df["shot_id"] = df["shot_id"].astype(str).apply(lambda x: re.sub(r"^\d+_", "", x))
df.to_csv(path, index=False)
print("fixed shot_id and overwrote:", path)