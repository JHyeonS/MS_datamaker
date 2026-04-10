📘 2. MS_datamaker_UTAH_2019/read.md
개요

이 모듈은 Utah FORGE 2019 DAS 데이터 (SGY)를 기반으로
event / noise / unlabel segment plan을 생성한다.

📂 데이터 구조
data/
  event/
    *.sgy
  noise/
    *.sgy
  unlabel/
    *.sgy
📥 입력
SGY raw files
event catalog (이미 매칭 완료된 상태 기준)
⚙️ 처리 단계
1. Inventory 생성
python scripts/make_inventory.py \
  --event_dir data/event \
  --noise_dir data/noise \
  --unlabel_dir data/unlabel \
  --out_csv outputs/inventory.csv \
  --original_fs 2000
2. Segment Plan 생성
event
python scripts/make_segment_plan_event_2s.py
noise / unlabel
python scripts/make_segment_plan_noise_unlabel_2s.py
📤 FINAL로 전달
outputs/
  inventory.csv
  segment_plan_event.csv
  segment_plan_noise.csv
  segment_plan_unlabel.csv

👉 MS_datamaker_FINAL/input/utah_2019/로 복사

⚠️ 주의사항
sampling rate: 2000 Hz (→ 이후 1000 Hz로 resample 가능)
lower / upper view 존재 가능
group_id = raw file 기준
event catalog 매칭 정확도가 중요