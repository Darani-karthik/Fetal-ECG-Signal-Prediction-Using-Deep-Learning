Test subject(s): r10  (never seen during training)

| Domain | Metric | Value |
|---|---|---|
| Scalogram | MSE | 0.0127 |
| Scalogram | RMSE | 0.1128 |
| Scalogram | MAE | 0.0463 |
| Waveform | Pearson correlation | 0.195 |
| Waveform | Median windowed correlation (2 s) | 0.331 |
| Waveform | SNR, best raw abdominal lead -> U-Net | 0.01 -> 0.17 dB |
| Waveform | SNR gain | +0.16 dB |
| Waveform | Correlation ceiling of the CWT/ICWT transform | 0.715 |

| Beat detection (+-50 ms) | Precision | Recall | F1 |
|---|---|---|---|
| U-Net output | 0.948 | 0.859 | 0.901 |
| Direct scalp ECG (detector sanity check) | 1.000 | 0.994 | 0.997 |
| Best raw abdominal lead (baseline) | 0.185 | 0.139 | 0.159 |
