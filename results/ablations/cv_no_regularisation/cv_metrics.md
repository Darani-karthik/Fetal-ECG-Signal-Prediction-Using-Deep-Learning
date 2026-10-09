| Test subject | Correlation | Median win. corr. | SNR gain (dB) | Beat F1 (U-Net) | Beat F1 (best raw lead) |
|---|---|---|---|---|---|
| r01 | 0.895 | 0.937 | +6.51 | 0.999 | 0.390 |
| r04 | 0.343 | 0.346 | +0.52 | 0.411 | 0.187 |
| r07 | 0.602 | 0.705 | +1.96 | 0.985 | 0.162 |
| r08 | 0.912 | 0.959 | +7.00 | 0.999 | 0.554 |
| r10 | 0.526 | 0.557 | +1.38 | 0.628 | 0.159 |
| **mean ± std** | **0.656 ± 0.219** | **0.701 ± 0.232** | **+3.47 ± 2.72** | **0.805 ± 0.243** | **0.290 ± 0.157** |

| Beat detection (mean ± std over subjects) | Precision | Recall | F1 |
|---|---|---|---|
| U-Net output | 0.964 ± 0.045 | 0.742 ± 0.311 | 0.805 ± 0.243 |
| Direct scalp ECG (detector sanity check) | 0.999 ± 0.002 | 0.995 ± 0.006 | 0.997 ± 0.004 |
| Best raw abdominal lead (baseline) | 0.352 ± 0.188 | 0.247 ± 0.135 | 0.290 ± 0.157 |
