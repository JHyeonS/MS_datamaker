# MS_datamaker_PH Process

## 1. 프로젝트 목표
본 프로젝트의 목표는 포항 Site-A DAS 데이터에서 microseismic event 후보를 추출하고,  
이를 이미지 및 이후 HDF5 데이터셋 형태로 정리하여 학습 가능한 데이터셋을 구축하는 것이다.

---

## 2. 원본 데이터 구조 파악
원본 DAS 데이터는 `/data2/Hyeonsu` 아래에 주 단위 폴더로 저장되어 있다.

예:
- `190522-190528`
- `190529-190604`
- `190923-190930`

각 폴더 내부에는 `.tdms` 파일이 존재하며, 파일명에 시간 정보가 포함되어 있다.

예:
- `1khz_50ns_20deci_2m_UTC_20190926_174200.000.tdms`
- `1khz_50ns_20deci_2m_UTC+0900_DST0_20190528_103859.387.tdms`

파일명으로부터 다음 정보를 읽을 수 있다.
- 샘플링레이트: `1khz`
- 시간대 정보: `UTC` 또는 `UTC+0900`
- 파일 시작 시각: `YYYYMMDD_HHMMSS.mmm`

---

## 3. TDMS 파일 인덱스 생성
### 목적
전체 TDMS 파일을 빠르게 검색하기 위해, 파일명 기반 메타 인덱스를 생성하였다.

### 사용 코드
- `scripts/make_tdms_index.py`

### 수행 내용
- `/data2/Hyeonsu` 이하의 모든 `.tdms` 파일을 탐색
- `filename`, `filepath`, `starttimeUTC`, `endtimeUTC`, `duration_sec` 정보를 CSV로 저장
- 이 단계에서는 **TDMS 내부를 읽지 않고 파일명만 파싱**하여 속도를 확보함

### 출력
- `outputs/tdms_file_index.csv`

---

## 4. Event catalog와 TDMS 교집합 찾기
### 목적
catalog에 기록된 microseismic event와 실제 TDMS 파일 시간 구간의 교집합을 찾는다.

### 사용 코드
- `scripts/match_catalog_tdms.py`

### 입력
- `csv/iDAS_detection_korean_microeqrthquake_list_20190501_20191001.xlsx`
- `outputs/tdms_file_index.csv`

### 수행 내용
- catalog의 event 시간을 읽음
- 각 event가 포함되는 TDMS 파일을 찾음
- 매칭 결과를 CSV로 저장

### 주의사항
초(second) 정보가 없는 catalog 항목이 많아서, 이 매칭은 **정확한 위상 위치 탐지**가 아니라  
**candidate TDMS file 선정** 수준으로 해석해야 한다.

### 출력
- `outputs/catalog_tdms_intersection.csv`

---

## 5. 매칭된 TDMS 파일 복사
### 목적
매칭된 이벤트와 관련된 TDMS 파일만 프로젝트 디렉토리로 복사하여 이후 작업을 단순화한다.

### 사용 코드
- `scripts/copy_matched_tdms.py`

### 수행 내용
- `catalog_tdms_intersection.csv`에서 `matched == True` 인 파일 선택
- 선택된 파일을 `data/raw_tdms/` 아래로 복사

### 출력 디렉토리
- `data/raw_tdms/`

---

## 6. 10초 단위 이미지 생성
### 목적
event의 정확한 초 단위 시각이 없기 때문에,  
우선 매칭된 TDMS 파일 전체를 10초 단위로 잘라 이미지로 저장한 뒤 직접 위상 출현 여부를 확인한다.

### 사용 코드
- `scripts/make_tdms_10s_images.py`

### 수행 내용
- 매칭된 TDMS 파일을 읽음
- 10초 단위로 segmentation
- 각 segment를 DAS image로 저장
- 이후 사용자가 직접 이미지들을 확인하여 위상 출현 구간을 선별

### 출력 디렉토리
- `outputs/tdms_10s_images/`

---

## 7. 현재 방법론의 의미
현재 catalog에는 분 단위 정보만 존재하여 exact event time 기반 자동 slicing이 어렵다.  
따라서 현재 단계에서는 다음 전략이 타당하다.

1. candidate TDMS file 선정
2. 10초 단위 segmentation
3. 이미지 직접 확인
4. 위상이 보이는 구간 수동/반자동 선별
5. 이후 HDF5 데이터셋 생성

즉 현재 단계는 **정밀 자동 추출 단계가 아니라 candidate screening 및 라벨 정제 단계**로 볼 수 있다.

---

## 8. 다음 단계
향후 진행 순서는 다음과 같다.

1. `outputs/tdms_10s_images/` 확인
2. 위상이 보이는 segment 목록 정리
3. 선택된 segment를 다시 waveform slice로 저장
4. HDF5 포맷으로 dataset 구축
5. 학습용 train/val/test split 구성

---

## 10. 기존 이벤트 이미지로부터 catalog 복원
### 목적
기존 연구자가 생성해둔 event 이미지 파일명에서 시간 정보를 추출하여,  
기존 workflow와 호환되는 event catalog를 복원한다.

### 사용 코드
- `scripts/make_catalog.py`

### 입력
- 기존 event 이미지 폴더들
  - `deep_svdd_test/1_event`
  - `das_image_old/1_event`
  - `das_image/1_event`
  - `1026_train/event`
  - `1026_test/1_event`
  - `1010_test/event`

### 수행 내용
- 각 `.png` 파일명에서 시간 정보 추출  
  - 예: `190917_184042_000.png`
- 중복 시간 이미지 제거
- 시간 간격이 가까운 이미지들을 하나의 cluster(event)로 묶음
- cluster의 시작 시각을 event 대표 시각으로 사용
- 이후 기존 `match_catalog_tdms.py`에서 사용할 수 있도록  
  `date`, `time`, `ms` 형식의 catalog CSV 생성
- 추가로 cluster 시작/종료시각, 길이, 대표 파일명 등을 담은 debug CSV도 생성

### 출력
- `recovered_catalog.csv`
- `recovered_catalog_debug.csv`

### 주의사항
이미지 파일명 시간 기준이 KST/UTC 중 무엇인지 반드시 확인해야 하며,  
이후 TDMS 매칭 workflow와 동일한 시간 기준으로 통일해야 한다.

---

## 11. 복원 catalog와 기존 지진 목록의 중복 검증
### 목적
복원한 catalog가 기존 지진 목록과 실제로 얼마나 겹치는지 확인한다.

### 사용 코드
- `scripts/check_csv_matching.py`

### 입력
- `csv/iDAS_detection_korean_microeqrthquake_list_20190501_20191001.xlsx`
- `recovered_catalog_debug.csv`

### 수행 내용
- 기존 Excel 지진 목록에서 event 시간 정보를 읽음
- 복원된 catalog의 cluster 시작 시각과 비교
- 일정 시간 허용 오차(tolerance) 내에 들어오는 event를 overlap으로 판단
- overlap 결과를 CSV로 저장

### 출력
- `event_overlap_check.csv`

### 의미
이 단계는 기존 연구자가 생성한 event 이미지 기반 catalog가  
기존 공인 지진 목록과 어느 정도 일치하는지 검증하는 단계이다.  
이를 통해 복원 catalog의 신뢰도와 시간 기준(UTC/KST)의 적절성을 점검할 수 있다.

---

## 12. 갱신된 디렉토리 구조 요약

```text
MS_datamaker_PH/
├── csv/
│   └── iDAS_detection_korean_microeqrthquake_list_20190501_20191001.xlsx
│
├── data/
│   └── raw_tdms/
│       └── (matched == True 인 TDMS 파일 복사본)
│
├── outputs/
│   ├── tdms_file_index.csv
│   ├── catalog_tdms_intersection.csv
│   ├── tdms_10s_images/
│   ├── recovered_catalog.csv
│   ├── recovered_catalog_debug.csv
│   └── event_overlap_check.csv
│
├── scripts/
│   ├── make_tdms_index.py
│   ├── match_catalog_tdms.py
│   ├── copy_matched_tdms.py
│   ├── make_tdms_10s_images.py
│   ├── make_catalog.py
│   └── check_csv_matching.py
│
├── src/
└── process.md