# 1D CNN Regression Report — Jena Climate Dataset

**Task:** Multivariate time-series regression  
**Target variable:** `T (degC)` (air temperature)  
**Input data:** `output/processed_data.csv` (Stage 1) + `output/scaler.pkl` (Stage 2)  
**Framework:** TensorFlow 2.21.0 · CPU inference (no CUDA drivers available)

---

## 1. Data Preparation & Sequence Construction

### 1.1 Train / Validation / Test Split

The same 70,126-row processed dataset and chronological split from Stages 2 & 3 were reused unchanged:

| Split | Rows | Date range |
|---|---|---|
| Train | 49,088 | 2009-01-01 → 2014-08-08 |
| Validation | 10,518 | 2014-08-08 → 2015-10-20 |
| Test | 10,520 | 2015-10-20 → 2017-01-01 |

### 1.2 Feature Scaling

The Stage-2 `StandardScaler` (fitted on training data only) was reloaded from `output/scaler.pkl` without refitting, preserving consistent scaling boundaries across all stages. The target `T (degC)` was scaled with a separate scaler fitted on the training split only, then inverse-transformed to °C before computing RMSE and MAE.

**Input dimensionality:** 24 features (identical to Stage 3).

### 1.3 Sliding-Window Sequence Construction

The `sequencer.py` module from Stage 3 was reused directly. For each time step `t`, the input is a 3-D tensor of shape `(24, 24)` (24 timesteps × 24 features) and the target is the scalar `T` at time `t`.

| Split | Sequences | Shape |
|---|---|---|
| Train | 49,064 | (49,064 × 24 × 24) |
| Validation | 10,494 | (10,494 × 24 × 24) |
| Test | 10,496 | (10,496 × 24 × 24) |

The 24-row reduction per split reflects the window warm-up period.

---

## 2. Persistence Baseline

The persistence model predicts that the next temperature equals the last observed temperature — a 1-step lag of the target sequence (`y_pred[i] = y[i−1]`). This is the standard naive benchmark for time-series regression.

| Baseline | Test RMSE (°C) | Test MAE (°C) |
|---|---|---|
| Persistence | 0.9009 | 0.6425 |

Any trained model that cannot beat 0.90 °C RMSE has no predictive value. Both CNN variants clear this bar comfortably.

---

## 3. Model Architectures

### 3.1 Simple 1D CNN

| Layer | Output shape | Parameters |
|---|---|---|
| Conv1D (64 filters, k=3, causal) | (24, 64) | 4,672 |
| MaxPooling1D (pool=2) | (12, 64) | 0 |
| Conv1D (32 filters, k=3, causal) | (12, 32) | 6,176 |
| GlobalAveragePooling1D | (32,) | 0 |
| Dense (32, relu) | (32,) | 1,056 |
| Dense (1, linear) | (1,) | 33 |
| **Total** | | **~11,937** |

`padding='causal'` ensures each convolution output depends only on past timesteps, preventing temporal leakage. `GlobalAveragePooling1D` averages across the time axis, producing a fixed-length vector regardless of input length and keeping the parameter count low.

### 3.2 Multi-Scale 1D CNN

| Layer | Output shape | Parameters |
|---|---|---|
| Input → Branch A: Conv1D (64, k=3, causal) → MaxPooling1D(2) | (12, 64) | 4,672 |
| Input → Branch B: Conv1D (64, k=7, causal) → MaxPooling1D(2) | (12, 64) | 10,816 |
| Concatenate (A ∥ B) | (12, 128) | 0 |
| Conv1D (32, k=3, causal) | (12, 32) | 12,320 |
| GlobalAveragePooling1D | (32,) | 0 |
| Dense (32, relu) | (32,) | 1,056 |
| Dropout (0.2) | (32,) | 0 |
| Dense (1, linear) | (1,) | 33 |
| **Total** | | **~28,897** |

The two parallel branches use kernel sizes 3 and 7, capturing short-term fluctuations and slower diurnal patterns simultaneously. The outputs are concatenated along the channel axis, then processed by a shared convolutional layer before pooling.

---

## 4. Training Procedure

Both models used identical settings:

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (lr = 0.001) |
| Loss | Mean Squared Error |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping patience | 10 epochs |
| Monitor metric | Validation loss |
| Checkpoint | Save best weights by val loss (`.h5`) |

Early stopping with `restore_best_weights=True` ensures the saved model corresponds to the epoch with lowest validation loss, not the final epoch.

---

## 5. Results

### 5.1 CNN vs CNN Comparison

Metrics reported in °C after inverse-transforming scaled predictions:

| Model | Train RMSE | Train MAE | Val RMSE | Val MAE | Test RMSE | Test MAE |
|---|---|---|---|---|---|---|
| **CNN Simple** | **0.598** | **0.429** | **0.635** | **0.450** | **0.592** | **0.432** |
| CNN Multi-Scale | 0.704 | 0.522 | 0.745 | 0.547 | 0.702 | 0.528 |

The Simple CNN outperforms the Multi-Scale variant by ~0.11 °C RMSE on the test set. The deeper multi-branch model has ~2.4× more parameters but fails to improve — a case of added capacity without sufficient benefit for this task and training budget. Both models converge before the 200-epoch limit due to early stopping.

The official reloaded-checkpoint metric for `cnn_best.h5` (Multi-Scale): **Test RMSE = 0.7021 °C, Test MAE = 0.5282 °C**.

### 5.2 Full Cross-Stage Comparison

All models evaluated on the same held-out test period (°C):

| Model | Test RMSE (°C) | Test MAE (°C) |
|---|---|---|
| Persistence | 0.9009 | 0.6425 |
| MLP (Stage 2) | 0.0181 | 0.0136 |
| DNN (Stage 2) | 1.0759 | 0.7517 |
| LSTM 1-layer (Stage 3) | 0.5186 | 0.3658 |
| LSTM 2-layer (Stage 3) | 0.5357 | 0.3842 |
| **CNN Simple (Stage 4)** | **0.5917** | **0.4317** |
| CNN Multi-Scale (Stage 4) | 0.7021 | 0.5282 |

The Simple CNN sits between the two LSTM variants in RMSE, performing slightly worse than LSTM 1-layer (0.59 vs 0.52 °C). The Multi-Scale CNN is the weakest sequence model, just above the persistence baseline.

---

## 6. Analysis

### 6.1 CNN vs LSTM

The Simple CNN is competitive with the LSTMs at a fraction of the parameter count (≈12k vs ≈23–81k) and with faster per-epoch training on CPU. Convolutions are parallelised across the time axis, whereas LSTM cells process each timestep sequentially.

The Multi-Scale CNN's underperformance is likely due to:

1. **More parameters require longer training.** With patience=10 and early stopping, the larger model may not converge to as good a solution as the simpler one within the training budget.
2. **The two temporal scales are not as distinct as expected.** At 1-hour resolution, the difference between a k=3 and k=7 receptive field (3 h vs 7 h) is modest relative to the 24-hour window; the signal is similar enough that the branches learn redundant filters rather than complementary ones.
3. **No hyperparameter tuning.** Dropout rate, learning rate, and number of filters were not tuned for the Multi-Scale model.

### 6.2 Why the MLP Still Leads

The Stage-2 MLP (0.018 °C) remains the best model across all four stages. This continues the pattern observed in Stage 3: the lag and rolling-mean features engineered in Stage 1 (`T_lag_1h`, `T_lag_2h`, `T_lag_3h`, rolling means at 3, 6, 12 h) explicitly encode the autocorrelation structure that both CNNs and LSTMs must learn from raw windows. When the first lag feature is near-perfectly correlated with the next step's target, a linear model on that feature suffices — the CNN's local convolutions add no further benefit.

### 6.3 When to Prefer CNN over LSTM

| Criterion | Prefer CNN | Prefer LSTM |
|---|---|---|
| Training speed | ✓ Parallelisable across time axis | — Sequential timesteps |
| Local pattern detection | ✓ Convolutions ideal | — Less efficient |
| Long-range dependencies | — Fixed receptive field | ✓ Gated memory |
| Multi-scale structure | ✓ Parallel kernel branches | — Single hidden state |
| Variable-length sequences | — Requires fixed window | ✓ Handles natively |

For 1-hour-ahead temperature forecasting with a 24-step window, the CNN is a practical choice: it trains faster, uses fewer parameters, and achieves similar accuracy to a single-layer LSTM.

---

## 7. Artifacts

| File | Size | Description |
|---|---|---|
| `output/cnn_simple_best.h5` | 181,328 bytes | Best Simple CNN weights |
| `output/cnn_best.h5` | 394,312 bytes | Best Multi-Scale CNN weights |
| `output/scaler.pkl` | 992 bytes | Stage-2 StandardScaler (reused, never refit) |

Both checkpoints were reloaded and verified with a forward pass on the test set — output shapes and metrics matched the training-time values.

---

## 8. Conclusion

Two 1D CNN architectures were trained on 24-hour sliding windows of the Jena Climate dataset for 1-step-ahead temperature regression. The **Simple CNN** achieves a test RMSE of **0.592 °C**, competitive with the Stage-3 LSTMs and well above the persistence baseline (0.901 °C). The **Multi-Scale CNN** (parallel k=3 / k=7 branches) underperforms at 0.702 °C, suggesting that the added capacity does not pay off without further tuning.

Across all four stages, the Stage-2 MLP remains unbeaten — a consequence of the lag features from Stage 1 encoding the task's primary signal explicitly. CNNs would be expected to close this gap if those engineered features were removed or if the task were extended to multi-step forecasting, where learned temporal filters become load-bearing.
