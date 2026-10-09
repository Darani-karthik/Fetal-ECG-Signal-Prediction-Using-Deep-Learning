| Test subject | Correlation | Median win. corr. | SNR gain (dB) | Beat F1 (U-Net) | Beat F1 (best raw lead) |
|---|---|---|---|---|---|
| r01 | 0.893 | 0.939 | +6.43 | 1.000 | 0.390 |
| r04 | 0.421 | 0.471 | +0.83 | 0.886 | 0.187 |
| r07 | 0.545 | 0.644 | +1.54 | 0.994 | 0.162 |
| r08 | 0.887 | 0.940 | +5.98 | 0.999 | 0.554 |
| r10 | 0.521 | 0.565 | +1.35 | 0.917 | 0.159 |
| **mean ± std** | **0.653 ± 0.198** | **0.712 ± 0.194** | **+3.23 ± 2.45** | **0.959 ± 0.048** | **0.290 ± 0.157** |

| Beat detection (mean ± std over subjects) | Precision | Recall | F1 |
|---|---|---|---|
| U-Net output | 0.974 ± 0.031 | 0.946 ± 0.065 | 0.959 ± 0.048 |
| Direct scalp ECG (detector sanity check) | 0.999 ± 0.002 | 0.995 ± 0.006 | 0.997 ± 0.004 |
| Best raw abdominal lead (baseline) | 0.352 ± 0.188 | 0.247 ± 0.135 | 0.290 ± 0.157 |
