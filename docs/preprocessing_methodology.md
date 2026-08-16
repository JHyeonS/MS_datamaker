# Preprocessing Methodology

## Purpose

This preprocessing pipeline was built to make the dataset fixed, reproducible, and comparable across sites before model training. The previous direction was to perform most preprocessing immediately before the GPU model. The current direction is different: raw windows are converted into preprocessed `.npy` samples offline, and the model reads those fixed samples through CSV split files.

The main goals are:

- avoid AGC-based amplitude distortion,
- remove slow trend/DC-like components with median detrending and frequency filtering,
- normalize signal scale with site-wise RMS normalization,
- keep the same sample identity and split structure across preprocessing sweeps,
- evaluate how much site-specific frequency content should be preserved.

## Common Pipeline

All three final sweeps use the same input sample set and the same experiment split structure.

Common preprocessing steps:

1. Load raw 2-second DAS segment from the segment plan.
2. Optionally resample to the sweep target sampling rate.
3. Apply median detrending.
4. Apply FFT-based frequency filtering.
5. Compute RMS statistics by site.
6. Normalize each site to target RMS.
7. Save the final preprocessed window as `.npy`.
8. Save metadata, value statistics, frequency energy statistics, and split CSV files.

Common parameters:

```yaml
preprocess_mode: median_filter_rms
agc: false
detrend: median
rms_scope: site
target_rms: 1.0
```

The RMS normalization is site-wise:

```text
x_normalized = x / rms(site) * target_rms
```

This means each site is normalized to approximately RMS 1.0 after detrending and filtering. This is not peak clipping. Large transient peaks can remain after RMS normalization.

## Dataset Composition

All three sweeps contain the same number of samples:

```text
total: 20,048

pohang:    3,383
utah_2019: 10,497
utah_2023: 6,168

event:   2,385
noise:   3,234
unlabel: 14,429
```

## Sweep Definitions

### Sweep 01

Directory:

```text
output_npy/final/sweep01_current_fs1000_rms1p0_phlp50_ut19lp200_ut23lp500_3site
```

Configuration:

```yaml
target_fs: 1000
target_rms: 1.0
rms_scope: site
filters:
  pohang:
    low_hz: null
    high_hz: 50
  utah_2019:
    low_hz: null
    high_hz: 200
  utah_2023:
    low_hz: null
    high_hz: 500
```

Interpretation:

Sweep 01 preserves the most site-specific bandwidth. Pohang is limited to 50 Hz, Utah 2019 keeps up to 200 Hz, and Utah 2023 keeps up to 500 Hz. This may preserve useful information, but it also keeps strong site-specific frequency differences.

Observed value behavior after RMS normalization:

```text
abs(value) > 7 samples: 10,583 / 20,048 = 52.8%
max_abs: 1916.69
```

This sweep has the largest fraction of high-amplitude outlier samples.

### Sweep 02

Directory:

```text
output_npy/final/sweep02_mid_fs1500_rms1p0_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275_3site
```

Configuration:

```yaml
target_fs: 1500
target_rms: 1.0
rms_scope: site
filters:
  pohang:
    low_hz: 1.5
    high_hz: 50
  utah_2019:
    low_hz: 1.5
    high_hz: 125
  utah_2023:
    low_hz: 1.5
    high_hz: 275
```

Interpretation:

Sweep 02 is the middle condition between site-specific preservation and common filtering. It removes very low-frequency drift below 1.5 Hz and reduces the high-frequency bandwidth compared with Sweep 01, but still leaves different high cutoffs for each site.

Observed value behavior after RMS normalization:

```text
abs(value) > 7 samples: 9,141 / 20,048 = 45.6%
max_abs: 1913.18
```

This sweep reduces outlier frequency compared with Sweep 01, but large peaks still remain.

### Sweep 03

Directory:

```text
output_npy/final/sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site
```

Configuration:

```yaml
target_fs: 2000
target_rms: 1.0
rms_scope: site
filters:
  pohang:
    low_hz: 3
    high_hz: 50
  utah_2019:
    low_hz: 3
    high_hz: 50
  utah_2023:
    low_hz: 3
    high_hz: 50
```

Interpretation:

Sweep 03 applies the same 3-50 Hz bandpass filter to all sites. This is closest to the old `0406` frequency philosophy and is the most domain-normalized condition. It removes low-frequency drift and suppresses site-specific high-frequency content.

Observed value behavior after RMS normalization:

```text
abs(value) > 7 samples: 5,428 / 20,048 = 27.1%
max_abs: 1683.57
```

This sweep has the lowest outlier fraction and the most consistent frequency condition across sites.

## Raw Frequency Analysis

Separate raw frequency analyses were generated from the original segment plans, not from the preprocessed sweep `.npy` files.

Raw analysis directories:

```text
output_npy/final/raw_frequency_analysis_pohang
output_npy/final/raw_frequency_analysis_utah_2019
output_npy/final/raw_frequency_analysis_utah_2023
```

Each directory contains:

```text
raw_frequency_energy_by_site.csv
raw_frequency_energy_by_site_label.csv
raw_frequency_energy_by_site_label.png
raw_frequency_summary.json
figures_detail/raw_frequency_log_no_dc.png
figures_detail/raw_frequency_band_summary.png
figures_detail/raw_frequency_cumulative.png
figures_detail/raw_frequency_band_summary.csv
```

Important raw frequency observations:

- Pohang event energy is concentrated around the 10-50 Hz range.
- Pohang noise/unlabel have visible DC/low-frequency components, but much of their total energy is spread across high-frequency bands.
- Utah 2019 and Utah 2023 have substantial energy above 250 Hz.
- A simple linear per-bin frequency plot can make broadly distributed high-frequency energy look small. For interpretation, use the log-scale, cumulative, and band-summary figures.

Pohang band summary:

```text
                event   noise   unlabel
0-1 Hz          0.002   0.058   0.080
1-3 Hz          0.001   0.003   0.004
3-10 Hz         0.043   0.011   0.012
10-50 Hz        0.472   0.062   0.071
50-100 Hz       0.091   0.081   0.089
100-250 Hz      0.130   0.276   0.277
250+ Hz         0.261   0.509   0.467
```

Utah 2019 band summary:

```text
                event   noise   unlabel
0-1 Hz          0.061   0.026   0.041
1-3 Hz          0.006   0.003   0.004
3-10 Hz         0.007   0.006   0.007
10-50 Hz        0.045   0.038   0.042
50-100 Hz       0.086   0.062   0.066
100-250 Hz      0.276   0.288   0.285
250+ Hz         0.517   0.578   0.556
```

Utah 2023 band summary:

```text
                event   noise   unlabel
0-1 Hz          0.001   0.001   0.001
1-3 Hz          0.001   0.001   0.001
3-10 Hz         0.004   0.002   0.003
10-50 Hz        0.018   0.013   0.018
50-100 Hz       0.033   0.021   0.047
100-250 Hz      0.364   0.228   0.652
250+ Hz         0.580   0.735   0.280
```

## Methodological Interpretation

The three sweeps form a frequency-domain ablation:

```text
Sweep 01: preserve site-specific bandwidth strongly
Sweep 02: intermediate condition
Sweep 03: enforce common 3-50 Hz band across all sites
```

For site-transfer learning, Sweep 03 is the most defensible primary condition because:

- all sites share the same filter range,
- raw site-specific high-frequency differences are suppressed,
- outlier fraction after RMS normalization is the lowest,
- the preprocessing is closest to the old `0406` frequency philosophy,
- the method is easy to describe and reproduce.

Recommended priority:

```text
1. Sweep 03
2. Sweep 02
3. Sweep 01
```

Sweep 01 and Sweep 02 should still be kept as ablation experiments. They answer whether preserving broader site-specific frequency content helps or hurts detection and transfer.

## Site Transfer Splits

The generated split structure already supports site transfer learning. No new `.npy` data extraction is required. Training should use the relevant CSV split directory.

Stage 3: one-site to one-site transfer

```text
stage3_pohang_to_utah_2019
stage3_pohang_to_utah_2023
stage3_utah_2019_to_pohang
stage3_utah_2019_to_utah_2023
stage3_utah_2023_to_pohang
stage3_utah_2023_to_utah_2019
```

Stage 4: leave-one-site-out

```text
stage4_leave_one_site_out_pohang
stage4_leave_one_site_out_utah_2019
stage4_leave_one_site_out_utah_2023
```

For example, Sweep 03 Pohang to Utah 2019 transfer should use:

```text
output_npy/final/sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site/metadata/experiments/stage3_pohang_to_utah_2019
```

The `.npy` data do not need to be regenerated for site transfer. Only the referenced CSV split directory changes.

## Output Files to Use

Main preprocessed datasets:

```text
output_npy/final/sweep01_current_fs1000_rms1p0_phlp50_ut19lp200_ut23lp500_3site
output_npy/final/sweep02_mid_fs1500_rms1p0_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275_3site
output_npy/final/sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site
```

Per-sweep metadata:

```text
metadata/all_samples.csv
metadata/build_summary.json
metadata/preprocess_final_stats.json
metadata/min_max_after_rms.json
metadata/frequency_energy_by_scope.csv
metadata/frequency_energy_by_site_label.csv
metadata/sample_value_exceedance_summary_abs_gt_7p0.json
metadata/experiments/
```

Preview and analysis figures:

```text
output_npy/final/png/
```

Raw frequency analyses:

```text
output_npy/final/raw_frequency_analysis_pohang
output_npy/final/raw_frequency_analysis_utah_2019
output_npy/final/raw_frequency_analysis_utah_2023
```
