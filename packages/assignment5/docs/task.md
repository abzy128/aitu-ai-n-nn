# Implementation Plan — Autoencoder: Unsupervised Pattern Recognition
**Dataset:** Weather Dataset (Jena) · **Task:** Anomaly detection + latent space exploration + pattern clustering
**Builds on:** `output/processed_data.csv` (Stage 1), `scaler.pkl` (Stage 2), sequences from Stage 3/4

---

## Adaptation Notes
The original assignment links denoising back to binary classification from Stage 4. Since our pipeline is regression-based, this stage is adapted to three unsupervised use cases that are meaningful for weather data:

| Original | Adapted |
|---|---|
| Denoise → re-run classifier, compare accuracy | Denoise → use reconstruction error as **anomaly score** |
| Binary classification comparison (clean / noisy / denoised) | **Latent space visualization** colored by season, hour, temperature |
| Single experiment | + **Weather pattern clustering** via K-Means on bottleneck vectors |

The denoising experiment itself is kept — it directly satisfies the assignment's core autoencoder requirement and motivates the anomaly detection application naturally.

---

## Project Structure

```
zadanie_5/
├── notebooks/
│   └── 05_autoencoder.ipynb
├── utils/
│   ├── __init__.py
│   ├── data_prep.py        # reuse from Stage 2/3
│   ├── sequencer.py        # reuse from Stage 3
│   ├── noise.py            # add_gaussian_noise()
│   ├── models.py           # build_conv_autoencoder(), extract_encoder()
│   └── evaluator.py        # reconstruction plots, anomaly flagging, cluster viz
├── output/
│   ├── processed_data.csv  # from Stage 1
│   ├── scaler.pkl          # from Stage 2 — reload only
│   ├── autoencoder.h5      # full autoencoder
│   └── encoder.h5          # encoder half only (for latent space use)
└── requirements.txt
```

---

## Step 1 — Data Preparation & Sequences
**File:** `utils/data_prep.py` + `utils/sequencer.py` · **Notebook:** Section 1

- Load `output/processed_data.csv`, parse datetime index, drop NaN rows
- Reload `scaler.pkl` — `.transform()` only, including target (same as Stages 3 & 4)
- Chronological split at identical 70/15/15 boundaries
- Build sliding windows: `window_size=24`, reuse `sequencer.py` directly
- **This time retain the datetime index of each window** — needed to color latent space plots by season/hour later
  - Store as `window_timestamps`: the timestamp of the last step in each window

---

## Step 2 — Add Gaussian Noise
**File:** `utils/noise.py` · **Notebook:** Section 2

- `add_gaussian_noise(X, std=0.1)` — add `N(0, std²)` noise to every feature of every timestep
- Apply to `X_test` only (train stays clean — the autoencoder learns from clean data)
- **Visualization:** plot one feature (e.g. `T (degC)`) for a single window: original vs noisy side by side
- Sanity check: compute mean per-feature SNR to confirm noise level is visible but not destructive

> `std=0.1` is appropriate since all features are scaled to ~N(0,1) — this adds roughly 10% of the typical signal variance.

---

## Step 3 — 1D Convolutional Autoencoder
**File:** `utils/models.py` · **Notebook:** Section 3

Architecture — symmetric encoder/decoder operating on `(window_size=24, n_features)` inputs:

```
── Encoder ──────────────────────────────────────────────
Input(24, n_features)
→ Conv1D(64, kernel_size=3, relu, padding='same')
→ MaxPooling1D(2)                     # → (12, 64)
→ Conv1D(32, kernel_size=3, relu, padding='same')
→ MaxPooling1D(2)                     # → (6, 32)
→ Conv1D(16, kernel_size=3, relu, padding='same')
→ GlobalAveragePooling1D()            # → (16,)  ← bottleneck

── Decoder ──────────────────────────────────────────────
→ RepeatVector(6)                     # → (6, 16)
→ Conv1D(16, kernel_size=3, relu, padding='same')
→ UpSampling1D(2)                     # → (12, 16)
→ Conv1D(32, kernel_size=3, relu, padding='same')
→ UpSampling1D(2)                     # → (24, 32)
→ Conv1D(n_features, kernel_size=3, linear, padding='same')   # → (24, n_features)
```

- Compile: `Adam(lr=1e-3)`, `loss=mse`
- **Training input and output are both `X_train` (clean → clean)** — standard autoencoder
- Fit with `EarlyStopping(patience=10, restore_best_weights=True)`
- Plot train vs val reconstruction loss

---

## Step 4 — Denoising & Reconstruction Quality
**Notebook:** Section 4

- Pass `X_test_noisy` through the trained autoencoder → `X_test_denoised`
- **Visualization** (required by assignment): for 3–5 randomly selected test windows, plot one feature across the 24 timesteps:
  - Line 1: original clean signal
  - Line 2: noisy signal
  - Line 3: autoencoder reconstruction
- Compute per-sample reconstruction MSE on:
  - Clean test: `mse(X_test, autoencoder(X_test))`
  - Noisy test: `mse(X_test, autoencoder(X_test_noisy))` — how well does it denoise?
- Report mean reconstruction error for both variants

---

## Step 5 — Anomaly Detection via Reconstruction Error
**File:** `utils/evaluator.py` · **Notebook:** Section 5

- Compute reconstruction MSE per window on the **full dataset** (train + val + test)
- **Anomaly threshold:** 95th percentile of train reconstruction errors
- Flag windows above threshold as anomalous — these are weather patterns the autoencoder found hardest to reconstruct
- **Time series anomaly plot:** plot `T (degC)` across the full date range, overlay flagged anomaly windows as red shading
- Inspect a few flagged windows manually — do they correspond to known weather events? (e.g. sudden temperature drops, storms, sensor issues in the raw data)
- Report: total anomaly count, anomaly rate (%), example timestamps

> This is the unsupervised equivalent of the classification evaluation in the original assignment — reconstruction error takes the place of accuracy/F1.

---

## Step 6 — Latent Space Exploration
**File:** `utils/models.py` + `utils/evaluator.py` · **Notebook:** Section 6

- Extract encoder half as a standalone model: `encoder = Model(inputs=autoencoder.input, outputs=bottleneck_layer)`
- Save as `output/encoder.h5`
- Encode all windows: `Z = encoder.predict(X_all)` → shape `(n_windows, 16)`
- Reduce to 2D with **PCA** (fast, interpretable) and optionally t-SNE (slower, non-linear)
- **Three scatter plots of the 2D latent space**, each colored by a different metadata variable derived from `window_timestamps`:
  1. **Season** — winter / spring / summer / autumn
  2. **Hour of day** — grouped as night / morning / afternoon / evening
  3. **Temperature value** — continuous colormap of `T (degC)` at the window's last timestep
- Expected finding: seasons should form visible clusters; hour-of-day gradients should be visible within seasons

---

## Step 7 — Weather Pattern Clustering
**Notebook:** Section 7

- Apply **K-Means** to `Z` (the 16-dim latent vectors, not the 2D projection)
- Choose K using the elbow method (plot inertia for K=2…10)
- Decode each cluster centroid back through the decoder to get a "typical pattern" for that cluster: shape `(24, n_features)`
- For each cluster, plot the reconstructed `T (degC)` profile across the 24-step window
- Label clusters with a descriptive interpretation (e.g. "cold night pattern", "warm afternoon pattern", "rapid temperature drop")
- Show cluster distribution over time: stacked area chart of cluster assignments across the full date range — do clusters correspond to seasons?

---

## Step 8 — Save Artifacts
**Notebook:** Section 8

- `output/autoencoder.h5` — full model (encoder + decoder)
- `output/encoder.h5` — encoder only
- Reload autoencoder, run one reconstruction on `X_test`, assert output shape matches input
- Print final mean reconstruction MSE and anomaly rate as summary

---

## Requirements

```
tensorflow
keras
pandas
numpy
matplotlib
seaborn
scikit-learn   # PCA, t-SNE, K-Means
jupyter
```

---

## Evaluation Checklist

- [ ] `scaler.pkl` reloaded, not refit; window timestamps retained
- [ ] Gaussian noise added only to test split; visual comparison shown
- [ ] 1D Conv Autoencoder: symmetric encoder/decoder, bottleneck of size 16
- [ ] Trained clean-to-clean (not noisy-to-clean) — standard autoencoder
- [ ] Reconstruction plots: original / noisy / denoised for 3+ windows
- [ ] Anomaly threshold set from train reconstruction error distribution
- [ ] Anomaly windows overlaid on full temperature time series
- [ ] Encoder extracted as standalone model, saved separately
- [ ] Latent space scatter colored by season, hour, and temperature
- [ ] K-Means clustering: elbow plot, cluster centroids decoded and plotted
- [ ] `autoencoder.h5` and `encoder.h5` saved and reload-verified
- [ ] Notebook fully re-runnable top to bottom
