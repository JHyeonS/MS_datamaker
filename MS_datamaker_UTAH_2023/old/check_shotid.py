import pandas as pd

meta = pd.read_csv("/home/ted1204/MS_datamaker_UTAH_2023/outputs_npy/utah_2023_2s_6split/metadata/metadata_raw.csv")
lbl  = pd.read_csv("/home/ted1204/MS_datamaker_UTAH_2023/outputs_npy/utah_2023_2s_6split/metadata/patch_labels_v2.csv")

# shot_id 정규화
meta["shot_id"] = meta["shot_id"].astype(str)
lbl["shot_id"] = lbl["shot_id"].astype(str)
lbl["shot_id"] = lbl["shot_id"].apply(lambda x: x if x.endswith(".sgy") else x + ".sgy")

meta["segment_index"] = meta["segment_index"].astype(int)
lbl["segment_index"] = lbl["segment_index"].astype(int)
meta["split_id"] = meta["split_id"].astype(int)
lbl["split_id"] = lbl["split_id"].astype(int)

meta_keys = set(zip(meta["shot_id"], meta["segment_index"], meta["split_id"]))
lbl_keys  = set(zip(lbl["shot_id"], lbl["segment_index"], lbl["split_id"]))

print("meta rows:", len(meta))
print("label rows:", len(lbl))
print("matched keys:", len(meta_keys & lbl_keys))

print("\n[meta sample keys]")
for x in list(meta_keys)[:10]:
    print(x)

print("\n[label sample keys]")
for x in list(lbl_keys)[:10]:
    print(x)