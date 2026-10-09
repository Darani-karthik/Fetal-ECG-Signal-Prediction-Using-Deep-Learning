# Methodology

This document explains, step by step, how the project turns a 1-D signal-separation problem into a
computer-vision problem, and where each step lives in the code. It follows the project report
("Fetal ECG Segmentation", Team B4) and calls out every place where the implementation had to
make a choice the report leaves open.

```
abdominal EDF ─► band-pass ─► resample ─► z-score ─► CWT ─► [8 × 64 × 512 image] ─► U-Net ─► [2 × 64 × 512]
                                                                                              │
          beats ◄─ R-peak detector ◄─ fetal ECG ◄─ inverse CWT ◄─ Hann overlap-add ◄─ fetal scalogram
```

## 1. Data  (`fecg/data/`)

| | |
|---|---|
| Source | PhysioNet *Abdominal and Direct Fetal ECG Database* (`adfecgdb` v1.0.0) |
| Subjects | 5 women in labour, 38–41 weeks of gestation (records `r01 r04 r07 r08 r10`) |
| Channels | 4 abdominal leads (`Abdomen_1-4`) + 1 direct fetal scalp lead (`Direct_1`) |
| Format | EDF+, 1 kHz, 16-bit, 300 s per record |
| Labels | expert fetal-QRS annotations (`*.edf.qrs`) |

* `edf.py` is a small dependency-free EDF reader (handles the EDF+ annotation pseudo-channel).
* `annotations.py` parses the binary WFDB annotation files.
* `download.py` downloads everything and verifies the published SHA-256 checksums.

The pairing is what makes the translation problem supervised: the *input* is the mixture seen
on the mother's abdomen, the *target* is what the electrode on the fetal scalp recorded at the
same time.

## 2. Pre-processing  (`fecg/data/preprocess.py`)

1. **Band-pass** (4th-order Butterworth, zero phase, `sosfiltfilt`) on all five channels.
2. **Down-sample** 1 kHz → 250 Hz (`resample_poly`, anti-aliased). Fetal ECG energy is well below
   100 Hz, and the 4× smaller images make training 16× cheaper.
3. **Normalise** each channel per record: subtract the median, divide by the standard deviation.

> **Deviation from the report: 2–100 Hz instead of 0.5–100 Hz.**
> The scalp lead carries slow baseline drift: across the five subjects 25–82 % (median 60 %) of its
> variance lies below 2 Hz (14–66 % below 1 Hz). That drift is a property of the electrode, not of
> the heart, and it cannot be predicted from the abdomen; the CWT band cannot represent the part below
> its lowest frequency either. Measured on the test subject, the correlation ceiling of the
> transform itself was only 0.72 with the report's 0.5 Hz cut-off, against 0.99 once the drift is
> removed (cut-off 2 Hz, safely below the 2.1–2.5 Hz fetal heart-rate fundamental). The README
> reports the effect on every metric. The old setting is still available:
> `--set data.bandpass=[0.5,100] cwt.fmin=1.0`.

## 3. Scalograms  (`fecg/transforms.py`, `fecg/data/dataset.py`)

The **Continuous Wavelet Transform** with a complex Morlet wavelet (`cmor1.5-1.0`) maps a signal to
a complex image `W[scale, time]`. We use 64 log-spaced scales covering 2–100 Hz, computed once on
the *whole record* (so windows have no edge artefacts) and then cut into 512-sample windows
(2.048 s ≈ 4–5 fetal beats) with a hop of 128 samples.

For every abdominal lead two layers are built, exactly as in the report:

| Layer | Definition |
|---|---|
| magnitude | `log(1 + abs(W))` – energy, log-compressed because the maternal QRS is much stronger than the fetal one |
| phase | `angle(W) / pi` ∈ [−1, 1] – timing and alignment |

4 leads × 2 layers = **8-channel input image of 64 × 512**.

**Target.** The report trains on the "clean fetal scalogram". A magnitude-only image cannot be
turned back into a waveform (the phase is lost), so the target is the fetal scalogram's
**real and imaginary parts** (2 channels), divided by their training-set standard deviation. The
magnitude `|W|` used for the figures and the MSE/RMSE/MAE image metrics is derived from them.

**Inverse transform.** `icwt` uses the delta-function reconstruction of Torrence & Compo (1998):
`x(t) ≈ κ · Σ_s Re W(s, t) / √s`. The single constant κ (it absorbs the wavelet's admissibility
constant and the log spacing) is calibrated numerically on band-limited noise so that the round
trip has unit gain; the round-trip correlation inside the band is > 0.99 (`tests/test_transforms.py`).
Because the reconstruction is a fixed linear map applied independently per time sample, it is also
differentiable and cheap, which the loss exploits (section 5).

## 4. U-Net  (`fecg/models/unet.py`)

A standard U-Net (Ronneberger et al., 2015): a symmetric encoder–decoder with skip connections.

```
input   8 × 64 × 512
enc 1  16 × 64 × 512   ┐ DoubleConv = (Conv3×3 → BatchNorm → ReLU → Dropout) → Conv3×3 → BatchNorm → ReLU
enc 2  32 × 32 × 256   │ ↓ 2×2 max-pool between stages
enc 3  64 × 16 × 128   │
enc 4 128 ×  8 ×  64   │
bottleneck 256 × 4 × 32┘
dec 4 128 ×  8 ×  64   ┐ ↑ 2×2 transposed conv, concatenate the encoder feature map of the
dec 3  64 × 16 × 128   │   same resolution (skip connection), DoubleConv
dec 2  32 × 32 × 256   │
dec 1  16 × 64 × 512   ┘
output  2 × 64 × 512   (1×1 conv; real + imaginary part of the fetal scalogram)
```

1.94 M parameters (`model.base_channels` = 16; 32 gives 7.76 M). The *encoder* gathers context (what is maternal, what is fetal); the *skip
connections* carry the fine time–frequency detail that the narrow fetal QRS needs; the *decoder*
rebuilds the clean image.

## 5. Training  (`fecg/engine.py`, `fecg/losses.py`)

Pairs of (noisy abdominal scalogram, clean fetal scalogram) are fed for **30 epochs** (the report
used 20; the smaller, regularised network converges more slowly) with AdamW, a one-cycle learning-rate
schedule and mixed precision. The best epoch on the validation subject is kept.

The loss is `MSE(Re,Im) + 0.5·MSE(|W|) + 1.0·MSE(waveform) + 1.0·(1 − Pearson(waveform))`:

* `MSE(Re,Im)` – plain scalogram regression, the objective described in the report;
* `MSE(|W|)` – sharpens the *visible* scalogram;
* `MSE(waveform)` – the inverse CWT of the prediction is compared to the true fetal waveform, so the
  network is optimised for what is ultimately recovered, not only for the image;
* `1 − Pearson(waveform)` – the same comparison, but invariant to amplitude, which cannot be inferred
  reliably from the abdomen (the amplitude of the scalp signal differs between subjects).

**Regularisation.** With three training subjects the network can memorise them (in the hardest
cross-validation fold of the un-regularised model the training loss fell to 0.08 while the loss on the
unseen validation subject stayed above 5), so the defaults add: channel dropout (0.2),
weight decay (0.01), a smaller U-Net, and three augmentations applied to the *complex* coefficients
(valid because the CWT is linear): random lead shuffling, random per-lead gain, and a polarity flip of
the whole (input, target) pair. The README's ablation table shows what each step bought.

## 6. Reconstruction and evaluation  (`fecg/inference.py`, `fecg/evaluate.py`)

**Reconstruction.** Overlapping window predictions (hop 128) are blended with a Hann taper into
one complex scalogram for the whole record, and *one* inverse CWT turns it into the fetal ECG
waveform. Blending in the scalogram domain avoids seams at window borders.

**Protocol – subject-wise.** The network never sees the test subject: train `r01 r04 r07`,
validate/select on `r08`, test on `r10`. Because five subjects are few, `scripts/cross_validate.py`
also runs *leave-one-subject-out* cross-validation (every subject is the test subject once). Splitting
random windows instead would put almost identical, 75 %-overlapping windows of the same recording into
both train and test and overstate performance.

**Metrics.**

| Domain | Metric | Definition |
|---|---|---|
| Scalogram | MSE, RMSE, MAE | on magnitude images scaled to [0, 1] with the 99.5th percentile of the training magnitudes |
| Waveform | Pearson correlation | predicted vs. true fetal ECG, whole record (first/last second trimmed) |
| Waveform | median windowed correlation | the same on non-overlapping 2 s windows (robust to a few bad windows) |
| Waveform | SNR gain | `SNR(output) − SNR(best raw abdominal lead)`, both after optimal rescaling, dB |
| Beats | Precision / Recall / F1 | R-peaks of the predicted waveform vs. expert annotations, ±50 ms tolerance, one-to-one matching |

Beat detection (`fecg/beats.py`) is an energy-envelope (Pan–Tompkins-style) detector. To show that
the detector and the annotation alignment are sound, the same detector is run on the true scalp ECG
(F1 ≥ 0.99 on every subject). As a baseline it is also run on the best raw abdominal lead, which
mostly finds *maternal* beats.

The *transform ceiling* reported next to the waveform metrics is the correlation obtained by
inverting the **true** fetal scalogram, i.e. the best any scalogram model could achieve through this
CWT/ICWT pair.

## 7. Limitations

* Five subjects, ≈ 25 minutes of recording in total: results across subjects vary (see the
  cross-validation table) and should not be read as a clinical validation.
* The reference is a *scalp* lead and the model sees *abdominal* leads: it learns a lead-to-lead
  translation, so fine morphology (P/T waves) is only partly recovered; QRS timing is recovered best.
* Electrode artefacts present in the scalp reference but not in the abdomen are (correctly) not
  predicted and show up as the largest errors.
* The detector is a simple, threshold-based one; a stronger detector would change absolute F1 for
  every signal, including the ground truth.
