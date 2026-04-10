📘 1. MS_datamaker_PH/read.md
개요

이 모듈은 Pohang DAS microseismic 데이터 (TDMS)를 기반으로
event / noise / unlabel segment plan을 생성하고 FINAL로 전달하기 위한 데이터 준비 파이프라인이다.

📂 데이터 구조
data/
  event/
    *.tdms
  noise/
    *.tdms
  unlabel/
    *.tdms
📥 입력
TDMS raw files
(선택) event label csv (이미 반영되어 있을 경우 생략 가능)
⚙️ 처리 단계
1. Inventory 생성
python scripts/make_inventory.py \
  --event_dir data/event \
  --noise_dir data/noise \
  --unlabel_dir data/unlabel \
  --out_csv outputs/inventory.csv

생성:

inventory.csv
2. Segment Plan 생성
event
python scripts/make_segment_plan_event_2s.py
noise / unlabel
python scripts/make_segment_plan_noise_unlabel_2s.py

생성:

segment_plan_event.csv
segment_plan_noise.csv
segment_plan_unlabel.csv
📤 FINAL로 전달할 파일
outputs/
  inventory.csv
  segment_plan_event.csv
  segment_plan_noise.csv
  segment_plan_unlabel.csv

👉 이 4개 파일을 MS_datamaker_FINAL/input/pohang/에 복사

⚠️ 주의사항
segment 길이는 2초 기준
channel 범위는 고정 또는 사전 정의
group_id는 동일 TDMS 파일 기준으로 유지
leakage 방지를 위해 동일 파일은 split되지 않도록 설계됨