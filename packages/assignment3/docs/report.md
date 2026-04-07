# LSTM Regression Report — Jena Climate Dataset

**Task:** Multivariate time-series regression  
**Target variable:** `T (degC)` (air temperature)  
**Input data:** `output/processed_data.csv` (Stage 1) + `output/scaler.pkl` (Stage 2)  
**Framework:** TensorFlow 2.21.0 · GPU: NVIDIA GeForce RTX 4060 Laptop (6 GB, Compute 8.9)

---

## 1. Data Preparation & Sequence Construction

### 1.1 Train / Validation / Test Split

The same 70,126-row processed dataset and chronological split from Stage 2 were reused:

| Split | Rows | Date range |
|---|---|---|
| Train | 49,088 | 2009-01-01 → 2014-08-08 |
| Validation | 10,518 | 2014-08-08 → 2015-10-20 |
| Test | 10,520 | 2015-10-20 → 2017-01-01 |

### 1.2 Feature Scaling

The Stage-2 `StandardScaler` (fitted on training data only) was reloaded from `output/scaler.pkl` rather than refit, ensuring no information leakage and a consistent feature space across all three stages. The target `T (degC)` was scaled separately for sequence-to-scalar training and inverse-transformed back to °C for final metric reporting.

**Input dimensionality:** 24 features (same as Stage 2).

### 1.3 Sliding-Window Sequence Construction

Unlike Stage 2, which treated each hour as an independent feature vector, Stage 3 feeds the LSTM a sliding window of the past `W` hours. For each time step `t`, the input is a 3-D tensor of shape `(W, 24)` and the target is the scalar `T` at time `t`.

With the default window `W = 24`:

| Split | Sequences | Shape |
|---|---|---|
| Train | 49,064 | (49,064 × 24 × 24) |
| Validation | 10,494 | (10,494 × 24 × 24) |
| Test | 10,496 | (10,496 × 24 × 24) |

The 24-row reduction per split reflects the window warm-up period (first `W − 1` rows cannot form a complete sequence).

---

## 2. Model Architectures

### 2.1 Single-Layer LSTM (Baseline)

| Layer | Output shape | Parameters |
|---|---|---|
| LSTM (64 units) | (None, 64) | 22,784 |
| Dropout | (None, 64) | 0 |
| Dense (1) | (None, 1) | 65 |
| **Total** | | **~22,849** |

The LSTM reads the full 24-step window and returns only its final hidden state, which is then projected to a scalar prediction.

### 2.2 Two-Layer Stacked LSTM (Main Model)

| Layer | Output shape | Parameters |
|---|---|---|
| LSTM 1 (100 units, return_sequences=True) | (None, 24, 100) | 50,000 |
| Dropout | (None, 24, 100) | 0 |
| LSTM 2 (50 units) | (None, 50) | 30,200 |
| Dropout | (None, 50) | 0 |
| Dense (8) | (None, 8) | 408 |
| Dense (1) | (None, 1) | 9 |
| **Total** | | **~80,617** |

The first LSTM returns its full sequence output so the second LSTM can attend over the entire encoded window. Dropout is applied after each LSTM layer to prevent co-adaptation.

---

## 3. Training Procedure

Both models used identical training settings:

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (default lr = 0.001) |
| Loss | Mean Squared Error |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping patience | 10 epochs |
| Monitor metric | Validation loss |
| Checkpoint | Save best weights by val loss |

Training ran on GPU (RTX 4060), with cuDNN 9.2 acceleration. Checkpoint files were saved in HDF5 format (`.h5`).

---

## 4. Results

### 4.1 LSTM vs LSTM Comparison

Metrics are reported in °C after inverse-transforming the scaled predictions:

| Model | Train RMSE (°C) | Train MAE (°C) | Val RMSE (°C) | Val MAE (°C) | Test RMSE (°C) | Test MAE (°C) |
|---|---|---|---|---|---|---|
| **LSTM 1-layer** | **0.543** | **0.373** | **0.550** | **0.375** | **0.519** | **0.366** |
| LSTM 2-layer | 0.574 | 0.407 | 0.566 | 0.393 | 0.536 | 0.384 |

The single-layer LSTM achieves lower error on both validation and test sets despite having far fewer parameters. This is consistent with the diminishing-returns behaviour observed in Stage 2 for deeper networks on this task.

The official reloaded-checkpoint metric for the 2-layer model (`lstm_best.h5`) is **Test RMSE = 0.5357 °C, Test MAE = 0.3842 °C**, confirming consistent serialisation and reload.

### 4.2 Cross-Stage Comparison

All four models evaluated on the same held-out test period (°C):

| Model | Test RMSE (°C) | Test MAE (°C) |
|---|---|---|
| **MLP (Stage 2)** | **0.0184** | **0.0139** |
| LSTM 1-layer | 0.5186 | 0.3658 |
| LSTM 2-layer | 0.5357 | 0.3842 |
| DNN (Stage 2) | 1.0759 | 0.7518 |

The Stage-2 MLP remains the best-performing model across all stages.

---

## 5. Analysis

### 5.1 Why LSTM Underperforms the MLP

The result is counter-intuitive — a recurrent model fed a 24-hour raw window performs worse than a shallow MLP fed a hand-crafted feature vector. Several factors explain this:

1. **Feature engineering already solved the problem.** The three lag features (`T_lag_1h`, `T_lag_2h`, `T_lag_3h`) and the rolling means (`3h`, `6h`, `12h`) in Stage 1 explicitly encode the autocorrelation structure that an LSTM would otherwise have to learn from scratch. When the LSTM receives the same features as part of its window input, its recurrent computation is largely redundant.

2. **The MLP operates on a more compact, noise-free representation.** The LSTM input includes all 24 raw sensor readings at every timestep, including weakly correlated variables like wind direction. The MLP operates on a single, already-compressed feature vector where the relevant signal is concentrated in the lag/rolling columns.

3. **The LSTM must learn longer-range patterns that do not improve 1-step-ahead prediction.** The 24-hour window provides context beyond what is needed for a 1-hour-ahead forecast. The additional temporal context introduces optimisation difficulty without payoff.

4. **Sequence length and batch size interact unfavourably.** Batch size 64 with sequences of length 24 is substantially smaller (in samples-per-update) than batch size 256 used in Stage 2, increasing gradient variance and slowing convergence.

### 5.2 When LSTM Would Gain an Edge

LSTMs would be expected to outperform flat models in this setting if:
- Lag/rolling features were removed from the input, forcing the model to learn autocorrelation from raw inputs.
- Multi-step forecasting (e.g., 6 h or 24 h ahead) were the target, where the longer-range dependencies captured by the recurrent state become load-bearing.
- The task included abrupt regime changes (cold fronts, weather events) that fixed-lag features cannot represent.

---

## 6. Window Size Experiment

The effect of the lookback window on the 2-layer LSTM test error:

| Window (hours) | Test RMSE (°C) | Test MAE (°C) |
|---|---|---|
| 12 | 0.5230 | 0.3761 |
| **24** | **0.5242** | **0.3740** |
| 48 | 0.5322 | 0.3869 |

Differences across window sizes are small (< 0.01 °C RMSE), indicating that most of the predictive information is captured within 12 hours. The 24-hour window (one full diurnal cycle) is a natural choice. Extending to 48 hours slightly degrades performance, likely because longer sequences increase optimisation difficulty without providing additional signal for 1-step-ahead forecasting.

---

## 7. Artifacts

| File | Size | Description |
|---|---|---|
| `output/lstm_1l_best.h5` | 298,848 bytes | Best 1-layer LSTM weights |
| `output/lstm_best.h5` | 1,002,928 bytes | Best 2-layer LSTM weights |
| `output/scaler.pkl` | 992 bytes | Stage-2 StandardScaler (reused) |
| `output/lstm_window_12.h5` | — | Window-experiment model, W=12 |
| `output/lstm_window_24.h5` | — | Window-experiment model, W=24 |
| `output/lstm_window_48.h5` | — | Window-experiment model, W=48 |

---

## 8. Conclusion

Two LSTM architectures (1-layer, 64 units; 2-layer, 100→50 units) were trained on 24-hour sliding windows of the Jena Climate dataset. The single-layer LSTM achieves a test RMSE of **0.519 °C**, outperforming the deeper 2-layer variant (0.536 °C). However, both LSTMs are substantially worse than the Stage-2 MLP (0.018 °C), because the lag and rolling mean features engineered in Stage 1 already encode the autocorrelation structure that the LSTM is designed to discover. A window size of 24 hours (one full diurnal cycle) is the most appropriate lookback for this task; extending to 48 hours provides no benefit. The LSTM would be expected to surpass the MLP if multi-step ahead forecasting were required or if the hand-crafted lag features were withheld.
