DAS Microseismic Dataset Creation Process
1. Raw Data Acquisition

Two DAS datasets are used to construct the microseismic detection dataset.

1.1 Pohang DAS Dataset

Source: Pohang EGS monitoring DAS system

Data format: TDMS

Sampling rate: site-dependent

Channels: selected DAS fiber segments

Labels created manually from visual inspection

Label distribution (approx):

Label	Count
noise	~750
event	~407
unlabeled	~2250
1.2 UTAH FORGE DAS Dataset

Source: UTAH FORGE DAS experiment

Data format: SEG-Y (.sgy)

Sampling rate: 2000 Hz

Channels: selected range from DAS fiber

Label distribution (approx):

Label	Count
noise	~900
event	~135
unlabeled	~3400
2. Raw DAS Data Processing

Raw DAS waveform data are converted into fixed-length segments.

2.1 Signal Preprocessing

Each DAS segment is processed with the following pipeline:

Bandpass filter

5 Hz – 80 Hz
Butterworth filter (order=4)

Robust normalization

(x - median) / MAD

Automatic Gain Control (AGC)

Window size:

~0.05 sec

Purpose:

equalize amplitude

enhance weak microseismic arrivals

2.2 Segment Generation

Raw waveform is divided into 1-second segments.

Segment characteristics:

shape: (channels, time)
time length: 1 sec
sampling rate: site dependent

Example:

2000 Hz sampling
→ 2000 samples per segment
3. Visual Inspection and Manual Labeling

Segments are visually inspected as DAS images.

Visualization pipeline:

bandpass
→ AGC
→ grayscale image
→ PNG visualization

Events are identified using:

coherent arrival pattern across channels

narrow spike-like phase

clear propagation signature

Ambiguous patterns such as:

stripe noise

incoherent fluctuations

are labeled as noise or unlabeled.

4. Dataset Storage Format

Segments are stored as NumPy arrays (.npy).

Example:

sample.npy

Array shape:

(C, T)

Where:

C = number of DAS channels
T = number of time samples

Example:

(400, 2000)
5. Metadata Construction

All samples are indexed in a metadata table.

File:

all_samples.csv

Example columns:

column	description
npy_path	path to .npy segment
site	pohang / utah
label	0=noise, 1=event, 2=unlabeled
label_name	noise/event/unlabel
group_id	original recording group
file_stem	source file identifier

Purpose:

reproducible dataset split

group-aware dataset management

6. Group-aware Dataset Split

Dataset is split using group-aware splitting to prevent leakage.

Split ratio:

train : val : test
= 0.8 : 0.1 : 0.1

Split key:

site + group_id

This ensures:

segments from the same DAS recording do not appear across splits.

7. Experimental Dataset Splits

Three experimental settings are constructed.

Stage 1 — Site-specific Experiments

Each site is trained and evaluated independently.

Example
stage1_pohang_only
stage1_utah_only

Split structure:

pretrain.csv
train.csv
val.csv
test.csv
Pretraining

Pretraining uses all non-test samples including:

noise
event
unlabeled
Fine-tuning

Fine-tuning uses labeled samples only:

noise
event
Stage 2 — Joint Training

Both sites are combined into a unified dataset.

Purpose:

evaluate multi-site learning

Structure:

stage2_joint
Stage 3 — Cross-site Generalization

Train on one site and evaluate on another.

Example:

train: Pohang
test: Utah

Purpose:

test generalization across DAS deployments
8. Class-balanced Split Constraints

Due to the rarity of microseismic events, additional constraints are applied during dataset splitting.

Minimum requirements:

train:
  event >= 20
  noise >= 100

val:
  event >= 10
  noise >= 30

test:
  event >= 10
  noise >= 30

The splitting algorithm searches for valid splits satisfying these constraints.

9. Dataset Quality Control

Random samples from each split are visualized to verify correctness.

Processing pipeline:

npy
→ bandpass
→ AGC
→ grayscale conversion
→ PNG visualization

This step confirms:

label correctness

absence of corrupted segments

consistency of DAS signal patterns.

10. Final Dataset Structure

Example directory layout:

outputs_npy
│
├─ samples
│   ├─ pohang
│   └─ utah
│
├─ metadata
│   ├─ all_samples.csv
│   └─ experiments
│        ├─ stage1_pohang_only
│        ├─ stage1_utah_only
│        ├─ stage2_joint
│        └─ stage3_cross_site

Each experiment folder contains:

pretrain.csv
train.csv
val.csv
test.csv
summary.json
11. Final Dataset Statistics (Approx)
Site	Noise	Event	Unlabeled
Pohang	~750	~407	~2250
Utah	~900	~135	~3400
12. Intended Machine Learning Pipeline

The dataset supports the following learning pipeline:

self-supervised pretraining
→ semi-supervised fine-tuning
→ microseismic detection

Pretraining leverages the large amount of unlabeled DAS data, while labeled noise/event samples are used for supervised fine-tuning.