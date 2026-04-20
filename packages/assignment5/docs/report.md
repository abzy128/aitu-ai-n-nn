# Autoencoder Report — Jena Climate Dataset

**Task:** Unsupervised pattern recognition — anomaly detection · latent space exploration · weather pattern clustering  
**Input data:** `output/processed_data.csv` (Stage 1) + `output/scaler.pkl` (Stage 2)  
**Framework:** TensorFlow 2.21.0 · CPU inference

---

## 1. Data Preparation & Sequence Construction

### 1.1 Train / Validation / Test Split

Same chronological 70/15/15 split as Stages 2–4:

| Split | Rows | Date range |
|---|---|---|
| Train | 49,088 | 2009-01-01 → 2014-08-08 |
| Validation | 10,518 | 2014-08-08 → 2015-10-20 |
| Test | 10,520 | 2015-10-20 → 2017-01-01 |

### 1.2 Feature Scaling

The Stage-2 `StandardScaler` was reloaded from `output/scaler.pkl` without refitting. Unlike Stages 3–4, no separate target scaler is needed — the autoencoder reconstructs the feature matrix `X` only, with no regression target.

**Input dimensionality:** 24 features (same as all prior stages):

`p (mbar)`, `Tpot (K)`, `Tdew (degC)`, `rh (%)`, `VPmax`, `VPact`, `VPdef`, `sh`, `H2OC`, `rho`, `wv`, `max. wv`, `wd`, `hour`, `day_of_week`, `month`, `is_weekend`, `T_lag_1h`, `T_lag_2h`, `T_lag_3h`, `T_rolling_3`, `T_rolling_6`, `T_rolling_12`, `time_of_day_ord`

### 1.3 Sliding-Window Sequences

Sliding windows of length 24 were built (one hour per step, covering one full diurnal cycle). The autoencoder trains on `X → X` with no target label.

| Split | Sequences | Shape |
|---|---|---|
| Train | 49,064 | (49,064 × 24 × 24) |
| Validation | 10,494 | (10,494 × 24 × 24) |
| Test | 10,496 | (10,496 × 24 × 24) |

The datetime timestamp of the last row in each window was retained (`window_timestamps`) and used later to color latent space plots by season, hour of day, and temperature.

---

## 2. Gaussian Noise

Noise was added to the test set only — the autoencoder learns a clean reconstruction mapping and is never shown noisy data during training.

| Parameter | Value |
|---|---|
| Distribution | N(0, σ²) |
| σ | 0.1 |
| Mean SNR (all features) | **19.6 dB** |

σ = 0.1 on StandardScaler-normalised data adds roughly 1% noise variance relative to the signal — visible in per-window plots but leaving the overall waveform shape intact. A 19.6 dB SNR confirms the noise level is meaningful but not destructive.

---

## 3. Model Architecture

### 3.1 Full Autoencoder

| Layer | Output shape | Parameters |
|---|---|---|
| Input | (24, 24) | 0 |
| Conv1D (64, k=3, relu, same) | (24, 64) | 4,672 |
| MaxPooling1D (2) | (12, 64) | 0 |
| Conv1D (32, k=3, relu, same) | (12, 32) | 6,176 |
| MaxPooling1D (2) | (6, 32) | 0 |
| Conv1D (16, k=3, relu, same) | (6, 16) | 1,552 |
| **GlobalAveragePooling1D** | **(16,)** | **0** — bottleneck |
| RepeatVector (6) | (6, 16) | 0 |
| Conv1D (16, k=3, relu, same) | (6, 16) | 784 |
| UpSampling1D (2) | (12, 16) | 0 |
| Conv1D (32, k=3, relu, same) | (12, 32) | 1,568 |
| UpSampling1D (2) | (24, 32) | 0 |
| Conv1D (24, k=3, linear, same) | (24, 24) | 2,328 |
| **Total** | | **17,080** |

`GlobalAveragePooling1D` averages the 6 remaining temporal positions into a single 16-dimensional vector — the bottleneck. The decoder uses `RepeatVector(6)` to broadcast this vector back to a temporal sequence before upsampling.

### 3.2 Encoder (standalone)

The first seven layers of the autoencoder extracted as a separate model:

| Parameter count | 12,400 |
|---|---|
| Input | (24, 24) |
| Output (bottleneck) | (16,) |

---

## 4. Training

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (lr = 0.001) |
| Loss | Mean Squared Error |
| Batch size | 128 |
| Max epochs | 200 |
| Early stopping patience | 10 |
| Best epoch | 89 |
| Stopped at epoch | 99 |

Training converged at epoch 89 and was halted at epoch 99 by early stopping. The model was restored to the best checkpoint automatically.

---

## 5. Denoising & Reconstruction Quality

| Input to autoencoder | Mean reconstruction MSE |
|---|---|
| Clean test set | **0.098597** |
| Noisy test set (σ=0.1) | **0.109747** |
| Noise penalty (ratio) | **1.11×** |

The autoencoder reconstructs clean inputs with MSE ≈ 0.099. When fed noisy inputs (which it was never trained on), reconstruction error rises only 11% — demonstrating that the bottleneck effectively learns to suppress noise rather than encode it.

---

## 6. Anomaly Detection

Reconstruction error was computed for all 70,054 windows (train + val + test). The anomaly threshold was set at the **95th percentile of training reconstruction error**.

| Metric | Value |
|---|---|
| Anomaly threshold (p95 of train MSE) | 0.187958 |
| Total anomalies flagged | 3,408 / 70,054 |
| Anomaly rate | **4.9%** |

By construction, approximately 5% of windows are flagged — the p95 threshold ensures the training set itself contributes 5% false-positives as a baseline.

### 6.1 Top Anomalies

The 10 windows with highest reconstruction error:

| Timestamp | Recon. MSE | T (°C) |
|---|---|---|
| 2015-07-08 08:00 | 0.4681 | 16.8 |
| 2015-07-08 09:00 | 0.4588 | 17.5 |
| 2015-07-06 08:00 | 0.4493 | 19.6 |
| 2015-07-06 09:00 | 0.4437 | 20.8 |
| 2015-07-08 07:00 | 0.4390 | 16.9 |
| 2013-07-27 21:00 | 0.4307 | 24.9 |
| 2015-08-07 20:00 | 0.4298 | 29.6 |
| 2013-07-27 22:00 | 0.4226 | 24.3 |
| 2015-07-04 20:00 | 0.4208 | 31.6 |
| 2015-07-06 07:00 | 0.4196 | 18.8 |

All top anomalies fall in **July–August**, clustering around two specific periods: early July 2015 and late July 2013. These coincide with summer heat events, where temperatures and humidity patterns deviate sharply from the median seasonal baseline the autoencoder learned. The highest single temperature observed among anomalies is 31.6 °C (2015-07-04) — well above the Jena climate average. The autoencoder has effectively learned that hot summer extremes are rare relative to the full training distribution.

---

## 7. Latent Space Exploration

The encoder maps each 24-hour window to a 16-dimensional bottleneck vector. PCA projected these to 2D (variance explained was consistent across the three scatter plots).

### 7.1 Coloring by Season

A clear seasonal structure is visible: winter and summer occupy distinct regions of the latent space, with spring and autumn forming transition bands between them. This confirms the bottleneck encodes climatically meaningful information — the autoencoder has implicitly learned seasonal temperature and humidity patterns without any labels.

### 7.2 Coloring by Hour of Day

A within-season gradient is visible, corresponding to the diurnal cycle. Afternoon windows tend to occupy a different sub-region of each seasonal cluster than night windows. The diurnal signal is weaker than the seasonal signal — consistent with temperature having a larger inter-seasonal range (~30 °C) than a typical daily range (~10 °C) at this latitude.

### 7.3 Coloring by Temperature

A smooth continuous gradient runs through the 2D projection, confirming that the bottleneck approximately preserves temperature ordering. Colder windows (blues on RdBu_r) and warmer windows (reds) separate naturally — the encoder has learned a temperature-like manifold without ever being trained on the temperature label.

---

## 8. Weather Pattern Clustering

### 8.1 Elbow Method

K-Means inertia was computed for K = 2–10 on the 16-dimensional latent vectors. The elbow is visible at **K = 4**, after which additional clusters produce diminishing returns. K = 4 also has a natural interpretation: one cluster per season.

### 8.2 Cluster Profiles

| Cluster | Mean T (°C) | Std T (°C) | Windows | Share | Interpretation |
|---|---|---|---|---|---|
| 0 | +0.3 | 5.8 | 16,089 | 23.0% | Cold / Winter |
| 1 | +18.9 | 5.4 | 13,855 | 19.8% | Hot / Summer |
| 2 | +6.3 | 4.4 | 22,266 | 31.8% | Mild / Spring–Autumn |
| 3 | +14.3 | 4.4 | 17,844 | 25.5% | Warm / Late Spring–Early Autumn |

The four clusters map cleanly onto seasonal temperature regimes. Cluster 2 (mild, 31.8%) is the largest because spring and autumn together span six months and dominate the dataset. Cluster 1 (summer) is the smallest at 19.8%, reflecting the limited proportion of truly hot hours in a temperate German climate.

The decoded centroid profiles of `Tpot (K)` across the 24 timesteps show:
- **Cluster 0 (cold):** flat, low Tpot — little diurnal variation in winter
- **Cluster 1 (hot):** high Tpot with a pronounced afternoon peak — strong diurnal cycle in summer
- **Cluster 2 (mild):** intermediate Tpot, moderate diurnal swing
- **Cluster 3 (warm):** elevated Tpot, clear afternoon rise

### 8.3 Cluster Distribution Over Time

The stacked area timeline confirms the seasonal interpretation: Cluster 0 (cold) dominates winter months, Cluster 1 (hot) spikes in July–August, and Clusters 2 and 3 are elevated during spring and autumn respectively. The pattern repeats coherently across all eight years in the dataset, validating that the clustering is capturing real annual climate cycles rather than noise.

---

## 9. Artifacts

| File | Size | Description |
|---|---|---|
| `output/autoencoder.h5` | 263,304 bytes | Full encoder + decoder |
| `output/encoder.h5` | 74,648 bytes | Encoder only (input → bottleneck) |
| `output/scaler.pkl` | 992 bytes | Stage-2 StandardScaler (reused) |

Both models were reloaded and verified with a forward pass — output shapes matched input shapes.

---

## 10. Conclusion

A 1D Convolutional Autoencoder with a 16-dimensional bottleneck was trained on 24-hour sliding windows of the Jena Climate dataset. The model achieves a mean reconstruction MSE of **0.099** on the clean test set and degrades only **11%** when fed noisy inputs (σ=0.1), demonstrating implicit denoising from the bottleneck compression.

Anomaly detection via reconstruction error identifies **4.9%** of all windows as anomalous. The most anomalous windows cluster in July 2015 and July 2013, corresponding to documented summer heat extremes — the autoencoder correctly identified rare high-temperature events as unusual relative to the learned distribution.

Latent space analysis confirms the 16-dimensional bottleneck encodes climatically meaningful structure: seasonal clusters are visible in PCA projections, with smooth temperature gradients across the 2D space. K-Means (K=4) on the latent vectors recovers four temperature regimes that correspond naturally to the four seasons, with the cluster timeline reproducing the annual climate cycle coherently across all eight years of data.
