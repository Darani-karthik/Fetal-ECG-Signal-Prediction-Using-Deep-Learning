| Test subject | Correlation | Median win. corr. | SNR gain (dB) | Beat F1 (U-Net) | Beat F1 (best raw lead) |
|---|---|---|---|---|---|
| r01 | 0.886 | 0.933 | +6.18 | 1.000 | 0.390 |
| r04 | 0.491 | 0.543 | +1.18 | 0.927 | 0.187 |
| r07 | 0.603 | 0.724 | +1.94 | 0.989 | 0.162 |
| r08 | 0.904 | 0.953 | +6.57 | 0.998 | 0.554 |
| r10 | 0.507 | 0.554 | +1.28 | 0.941 | 0.159 |
| **mean ± std** | **0.678 ± 0.181** | **0.741 ± 0.177** | **+3.43 ± 2.42** | **0.971 ± 0.031** | **0.290 ± 0.157** |

| Beat detection (mean ± std over subjects) | Precision | Recall | F1 |
|---|---|---|---|
| U-Net output | 0.978 ± 0.025 | 0.964 ± 0.037 | 0.971 ± 0.031 |
| Direct scalp ECG (detector sanity check) | 0.999 ± 0.002 | 0.995 ± 0.006 | 0.997 ± 0.004 |
| Best raw abdominal lead (baseline) | 0.352 ± 0.188 | 0.247 ± 0.135 | 0.290 ± 0.157 |
