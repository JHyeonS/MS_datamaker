📘 3. MS_datamaker_UTAH_2023/read.md
개요

이 모듈은 Utah FORGE 2023 DAS 데이터 (SGY)에서
하나의 raw 파일 내에 event / noise / unlabel이 혼재된 구조를 처리한다.

👉 이 점이 Pohang / Utah 2019와 가장 큰 차이

📂 데이터 구조
data/
  event_2417/
    RAW_*.sgy
📥 입력
SGY raw files
utah_2023_labels_2s.csv
(label csv 필수)
- file_name
- start_sec
- end_sec
- ch_start
- ch_end
- label
⚙️ 처리 단계
1. Inventory 생성
python scripts/make_inventory_utah2023.py \
  --raw_dir data \
  --out_csv outputs/inventory.csv \
  --ch_start 400 \
  --ch_end 805 \
  --original_fs 1000
2. Segment Plan 생성 (핵심)
python scripts/make_segment_plan_from_labels_utah2023.py \
  --inventory_csv outputs/inventory.csv \
  --label_csv csv/utah_2023_labels_2s.csv \
  --out_event_csv outputs/segment_plan_event.csv \
  --out_noise_csv outputs/segment_plan_noise.csv \
  --out_unlabel_csv outputs/segment_plan_unlabel.csv \
  --out_all_csv outputs/segment_plan_all.csv
📤 FINAL로 전달
outputs/
  inventory.csv
  segment_plan_event.csv
  segment_plan_noise.csv
  segment_plan_unlabel.csv

👉 MS_datamaker_FINAL/input/utah_2023/로 복사