# README: MS_datamaker_FINAL 입력 규격 정리

이 문서는 `MS_datamaker_Pohang`, `MS_datamaker_UTAH_2019`, `MS_datamaker_UTAH_2023`에서 최종 병합용 프로젝트인 `MS_datamaker_FINAL`에 **무엇을 제공해야 하는지**를 정리한 문서이다.

핵심 원칙은 단순하다.

- 각 개별 프로젝트는 자기 도메인의 원본 데이터를 직접 처리한다.
- `MS_datamaker_FINAL`은 원본 raw 데이터를 처음부터 해석하는 곳이 아니라,
  **각 도메인에서 정리한 메타데이터를 받아 공통 규격으로 병합하고 최종 dataset을 생성하는 곳**이다.
- 따라서 각 도메인 프로젝트는 최소한 아래 두 종류의 정보를 `MS_datamaker_FINAL`에 넘겨줘야 한다.

---

## 1. 전체 구조 요약

각 도메인 프로젝트가 `MS_datamaker_FINAL`에 제공해야 하는 것은 크게 두 가지다.

### 1) inventory 정보
원본 파일 목록과 파일별 속성

### 2) event label 정보
이벤트가 어느 파일의 어느 시간대에 존재하는지에 대한 정보

추가로, noise / unlabel이 존재한다면 이는 별도 label CSV가 아니라 **inventory 내의 `data_type` 컬럼**으로 구분하는 것을 기본 원칙으로 한다.

즉, 최종적으로 `MS_datamaker_FINAL`이 필요로 하는 입력은 다음과 같다.

- `inventory.csv`
- `event_labels.csv` (event가 있는 경우)

도메인별로 이 두 파일을 만들 수 있으면, `MS_datamaker_FINAL`에서:

1. inventory 병합
2. event label 병합
3. segment plan 생성
4. npy 생성
5. train/val/test split 생성

을 공통 방식으로 수행할 수 있다.

---

## 2. 각 프로젝트가 제공해야 하는 정확한 산출물

각 프로젝트는 아래 경로 형태로 파일을 넘겨주는 것을 권장한다.

```text
MS_datamaker_Pohang/
  export_to_final/
    inventory.csv
    event_labels.csv

MS_datamaker_UTAH_2019/
  export_to_final/
    inventory.csv
    event_labels.csv

MS_datamaker_UTAH_2023/
  export_to_final/
    inventory.csv
    event_labels.csv
```

이때 `event_labels.csv`는 event가 존재하는 도메인에 대해서만 필요하다.
noise / unlabel만 있는 데이터라면 inventory만 있어도 된다.

---

## 3. inventory.csv에서 반드시 포함해야 하는 컬럼

`inventory.csv`는 원본 파일 단위의 목록표다.

필수 컬럼은 아래와 같다.

| 컬럼명 | 설명 |
|---|---|
| `dataset_id` | 원본 데이터셋 이름. 예: `pohang`, `utah_2019`, `utah_2023` |
| `site` | 학습/평가 도메인 이름. 원칙적으로 `dataset_id`와 동일하게 두는 것을 권장 |
| `view` | 같은 raw 파일에서 여러 채널 범위를 따로 쓰는 경우의 식별자. 예: `pohang`, `utah_2019_1`, `utah_2019_2`, `utah_2023_main` |
| `data_type` | `event`, `noise`, `unlabel` 중 하나 |
| `file_path` | 실제 raw file 절대경로 |
| `file_name` | 파일명 |
| `file_stem` | 확장자를 뺀 파일명 |
| `ext` | 확장자. 예: `.tdms`, `.sgy`, `.segy` |
| `group_id` | split leakage 방지를 위한 그룹 식별자 |
| `original_fs` | 원본 샘플링 주파수 |
| `ch_start` | 사용할 시작 채널 인덱스 |
| `ch_end` | 사용할 끝 채널 인덱스 (exclusive) |

권장 추가 컬럼은 아래와 같다.

| 컬럼명 | 설명 |
|---|---|
| `parent_dir` | 원본 파일 상위 디렉토리 이름 |
| `raw_uid` | 파일명 충돌 방지를 위한 고유 식별자 |
| `notes` | 도메인별 특이사항 기록 |

### inventory 예시

```csv
dataset_id,site,view,data_type,file_path,file_name,file_stem,ext,group_id,original_fs,ch_start,ch_end
pohang,pohang,pohang,event,/data/pohang/event/A001.tdms,A001.tdms,A001,.tdms,pohang__A001,1000,243,648
pohang,pohang,pohang,noise,/data/pohang/noise/N003.tdms,N003.tdms,N003,.tdms,pohang__N003,1000,243,648
utah_2019,utah_2019,utah_2019_1,event,/data/utah2019/event/U101.sgy,U101.sgy,U101,.sgy,utah2019__U101,1000,679,1084
utah_2019,utah_2019,utah_2019_2,event,/data/utah2019/event/U101.sgy,U101.sgy,U101,.sgy,utah2019__U101,1000,273,678
utah_2023,utah_2023,utah_2023_main,event,/data/utah2023/event/Z220.tdms,Z220.tdms,Z220,.tdms,utah2023__Z220,1000,400,805
```

---

## 4. event_labels.csv에서 반드시 포함해야 하는 컬럼

`event_labels.csv`는 이벤트 시간 정보다.

필수 컬럼은 아래와 같다.

| 컬럼명 | 설명 |
|---|---|
| `dataset_id` | 원본 데이터셋 이름 |
| `site` | 도메인 이름 |
| `view` | inventory의 `view`와 연결되는 값 |
| `file_key` | inventory의 `file_name` 또는 매칭 가능한 파일 식별자 |
| `is_event` | event 여부. 보통 `1` |
| `label_source` | 라벨 출처 파일명 또는 메타정보 |

시간 정보는 아래 둘 중 하나가 필요하다.

### 방식 A: start/end 기반
| 컬럼명 | 설명 |
|---|---|
| `start_sec` | 이벤트 시작 시각(초) |
| `end_sec` | 이벤트 종료 시각(초) |

### 방식 B: center 기반
| 컬럼명 | 설명 |
|---|---|
| `center_sec` | 이벤트 중심 시각(초) |

권장 추가 컬럼:

| 컬럼명 | 설명 |
|---|---|
| `event_id` | 이벤트 고유 번호 |
| `confidence` | 라벨 신뢰도 |
| `annotator` | 라벨 작성자 |

### event label 예시 1: start/end 기반

```csv
dataset_id,site,view,file_key,start_sec,end_sec,center_sec,is_event,label_source
pohang,pohang,pohang,A001.tdms,12.0,14.0,,1,pohang_labels_2s.csv
utah_2019,utah_2019,utah_2019_1,U101.sgy,33.5,35.5,,1,utah_lower_labels_2s.csv
utah_2019,utah_2019,utah_2019_2,U101.sgy,33.5,35.5,,1,utah_upper_labels_2s.csv
```

### event label 예시 2: center 기반

```csv
dataset_id,site,view,file_key,start_sec,end_sec,center_sec,is_event,label_source
utah_2023,utah_2023,utah_2023_main,Z220.tdms,,,101.2,1,utah_2023_labels_2s.csv
```

---

## 5. group_id를 어떻게 정해야 하는가

`group_id`는 매우 중요하다. train/val/test split 시 **같은 원본에서 파생된 샘플이 서로 다른 split으로 흩어지는 것**을 막기 위해 필요하다.

기본 원칙은 아래와 같다.

- 같은 raw file에서 나온 샘플이면 같은 `group_id`
- 같은 event에서 여러 split / crop이 나온다면 가능하면 같은 `group_id`
- Utah 2023처럼 한 원본에서 다수 segment가 나오는 구조면 `group_id`를 더 보수적으로 묶는다

권장 규칙:

```text
group_id = {dataset_id}__{raw_file_stem}
```

Utah 2023처럼 더 세밀한 관리가 필요하면:

```text
group_id = {dataset_id}__{raw_file_stem}__{event_id}
```

---

## 6. view는 언제 필요한가

`view`는 같은 raw file에 대해 서로 다른 채널 범위를 사용해야 할 때 필요하다.

예를 들어 Utah 2019에서 lower / upper 두 구간을 따로 쓰는 경우:

- `utah_2019_1`
- `utah_2019_2`

이 둘은 `file_path`는 같을 수 있지만 `ch_start`, `ch_end`가 다르므로 inventory에서 별도 row로 관리해야 한다.

반대로 Pohang처럼 하나의 채널 범위만 고정 사용한다면:

- `view = pohang`

처럼 단일 값으로 두면 된다.

---

## 7. 각 개별 프로젝트가 해야 할 일

## 7-1. MS_datamaker_Pohang

제공해야 하는 것:

1. `inventory.csv`
   - event/noise/unlabel raw file 목록
   - Pohang의 고정 channel range 포함
   - Pohang 원본 fs 포함

2. `event_labels.csv`
   - 기존 `pohang_labels_2s.csv`를 공통 schema로 정규화한 것

즉, Pohang은 일반적으로 아래 3종 raw pool을 inventory에 넣어야 한다.

- event raw
- noise raw
- unlabel raw

그리고 event에 대해서만 label CSV를 별도로 제공한다.

---

## 7-2. MS_datamaker_UTAH_2019

제공해야 하는 것:

1. `inventory.csv`
   - event/noise/unlabel raw file 목록
   - lower/upper view를 각각 별도 row로 포함
   - Utah 2019의 fs 포함
   - 각 view별 channel range 포함

2. `event_labels.csv`
   - 기존 lower/upper label CSV를 공통 schema로 정규화한 것
   - 또는 lower/upper를 각각 제공해도 되지만, `MS_datamaker_FINAL`에 넘기기 전에는 하나로 합쳐두는 것이 더 좋다

중요:

- 같은 raw file이라도 lower/upper view는 inventory에서 두 row가 필요하다
- 하지만 `group_id`는 동일하게 유지하는 것이 일반적으로 적절하다

---

## 7-3. MS_datamaker_UTAH_2023

제공해야 하는 것:

1. `inventory.csv`
   - 가능한 한 raw file 기준으로 작성
   - event/noise/unlabel 여부를 `data_type`으로 명시
   - 원본 fs와 channel range 포함
   - split leakage 방지를 위한 `group_id` 설계 포함

2. `event_labels.csv`
   - Utah 2023 이벤트 시간 정보를 공통 schema로 정리
   - start/end 기반이면 그대로 사용
   - center 기반이면 `center_sec` 사용

Utah 2023은 현재 가장 주의가 필요한 도메인이다.

이유:
- raw 구조가 기존과 다를 수 있음
- 이미 잘린 segment / split 기반 산출물이 섞여 있을 수 있음
- 같은 원본에서 여러 파생 샘플이 생길 수 있음

따라서 Utah 2023은 가능하면 반드시 **raw file 기준 inventory**를 먼저 만들어야 한다.

이미 잘린 npy만 있고 raw가 없다면 차선책으로 inventory를 만들 수는 있지만, 그 경우에도 최소한 아래 정보는 복원해야 한다.

- 원래 어떤 raw에서 왔는지
- 같은 event / same raw끼리 어떤 group으로 묶어야 하는지
- 사용하는 채널 범위가 무엇인지
- 원래 fs가 무엇인지

---

## 8. MS_datamaker_FINAL이 이 파일들로 하는 일

`MS_datamaker_FINAL`은 각 프로젝트가 제공한 `inventory.csv`, `event_labels.csv`를 받아 다음 순서로 처리하면 된다.

### Step 1. inventory 병합
각 도메인의 inventory를 concat하여 `inventory_all.csv` 생성

### Step 2. event label 병합
각 도메인의 event label을 concat하여 `event_labels_all.csv` 생성

### Step 3. event segment plan 생성
`inventory_all.csv`와 `event_labels_all.csv`를 매칭하여 `segment_plan_event.csv` 생성

### Step 4. noise / unlabel segment plan 생성
`inventory_all.csv`에서 `data_type`이 `noise`, `unlabel`인 raw 파일을 기반으로 랜덤 segment 추출

### Step 5. npy build
segment plan CSV를 기반으로 최종 `.npy` 생성

### Step 6. all_samples.csv 및 split 생성
학습용 manifest와 실험 split 생성

---

## 9. 최종 체크리스트

각 프로젝트는 `MS_datamaker_FINAL`에 넘기기 전에 아래를 확인해야 한다.

### 공통 체크리스트

- [ ] `inventory.csv`가 존재한다
- [ ] `event_labels.csv`가 존재한다 (event가 있는 경우)
- [ ] `file_path`가 실제 존재하는 raw file을 가리킨다
- [ ] `file_name`, `file_stem`이 올바르다
- [ ] `dataset_id`, `site`, `view` naming이 일관적이다
- [ ] `group_id`가 split leakage를 막도록 설계되어 있다
- [ ] `original_fs`가 정확하다
- [ ] `ch_start`, `ch_end`가 정확하다
- [ ] `event_labels.csv`의 `file_key`가 inventory의 file과 매칭 가능하다

### Utah 2019 추가 체크

- [ ] lower / upper view가 inventory에 각각 들어가 있다
- [ ] lower / upper의 channel range가 정확하다

### Utah 2023 추가 체크

- [ ] raw file 기준 inventory인지 확인했다
- [ ] 같은 raw/event에서 파생된 샘플이 같은 `group_id`로 묶인다
- [ ] label 시간 정보가 실제 raw time axis와 일치한다

---

## 10. 한 줄 결론

각 개별 프로젝트가 `MS_datamaker_FINAL`에 넘겨야 하는 것은 단순히 label CSV만이 아니다.

**반드시 아래 두 가지를 함께 제공해야 한다.**

1. `inventory.csv`  
   → 무엇을 자를지 정의

2. `event_labels.csv`  
   → 어디를 자를지 정의

이 두 파일이 있어야 `MS_datamaker_FINAL`에서 공통 규격으로 segment plan을 만들고, 최종 병합 dataset을 안정적으로 생성할 수 있다.
