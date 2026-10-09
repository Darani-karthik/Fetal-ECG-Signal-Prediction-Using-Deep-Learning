Test subject(s): r10  (never seen during training)

| Domain | Metric | Value |
|---|---|---|
| Scalogram | MSE | 0.0100 |
| Scalogram | RMSE | 0.1001 |
| Scalogram | MAE | 0.0523 |
| Waveform | Pearson correlation | 0.516 |
| Waveform | Median windowed correlation (2 s) | 0.544 |
| Waveform | SNR, best raw abdominal lead -> U-Net | 0.04 -> 1.36 dB |
| Waveform | SNR gain | +1.32 dB |
| Waveform | Correlation ceiling of the CWT/ICWT transform | 0.993 |

| Beat detection (+-50 ms) | Precision | Recall | F1 |
|---|---|---|---|
| U-Net output | 0.942 | 0.921 | 0.931 |
| Direct scalp ECG (detector sanity check) | 1.000 | 0.994 | 0.997 |
| Best raw abdominal lead (baseline) | 0.185 | 0.139 | 0.159 |
