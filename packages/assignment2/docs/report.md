# MLP & DNN Regression Report — Jena Climate Dataset

**Task:** Multivariate time-series regression  
**Target variable:** `T (degC)` (air temperature)  
**Input data:** `output/processed_data.csv` (Stage 1 EDA output)  
**Framework:** TensorFlow 2.21.0 (CPU)

---

## 1. Data Preparation

### 1.1 Train / Validation / Test Split

The 70,126-row processed dataset was split chronologically (no shuffling) to prevent data leakage from future observations into training:

| Split | Rows | Date range |
|---|---|---|
| Train | 49,088 (70%) | 2009-01-01 → 2014-08-08 |
| Validation | 10,518 (15%) | 2014-08-08 → 2015-10-20 |
| Test | 10,520 (15%) | 2015-10-20 → 2017-01-01 |

### 1.2 Feature Scaling

The target column `T (degC)` was excluded from the feature matrix. The 24-feature input was standardised using `StandardScaler` fitted exclusively on the training split and applied to all three splits. The fitted scaler was saved to `output/scaler.pkl` for inference use.

**Features used (24):** atmospheric pressure, potential temperature, dew point, relative humidity, saturation/actual/deficit vapor pressure, specific humidity, water vapor concentration, air density, wind velocity, max wind velocity, wind direction, hour, day-of-week, month, is-weekend, T lag 1h/2h/3h, T rolling mean 3h/6h/12h, time-of-day ordinal.

---

## 2. Model Architectures

### 2.1 MLP (Shallow Baseline)

A compact 3-hidden-layer feedforward network:

| Layer | Units | Activation |
|---|---|---|
| Dense 1 | 64 | ReLU |
| Dense 2 | 32 | ReLU |
| Dense 3 | 16 | ReLU |
| Output | 1 | Linear |

- **Total parameters:** 4,225 (~16.5 KB)
- No regularisation layers; relies on early stopping for generalisation.

### 2.2 DNN (Deep Network)

A 6-hidden-layer network with interleaved Dropout for regularisation:

| Layer | Units | Activation | Dropout |
|---|---|---|---|
| Dense 1 | 128 | ReLU | 0.x |
| Dense 2 | 64 | ReLU | 0.x |
| Dense 3 | 32 | ReLU | 0.x |
| Dense 4 | 16 | ReLU | 0.x |
| Dense 5 | 8 | ReLU | 0.x |
| Output | 1 | Linear | — |

- **Total parameters:** 14,209 (~55.5 KB)
- Dropout applied after each hidden layer to prevent co-adaptation of neurons.

---

## 3. Training Procedure

Both models were trained with the same configuration:

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (default lr) |
| Loss | Mean Squared Error |
| Batch size | 256 |
| Max epochs | 200 |
| Early stopping patience | 10 epochs |
| Monitor metric | Validation loss |
| Checkpoint | Save best weights by val loss |

Training was conducted on CPU (no CUDA-capable GPU available). The best model weights were saved to `output/mlp_best.keras` (81 KB) and `output/dnn_best.keras` (222 KB). Early stopping ensured neither model trained for the full 200 epochs.

---

## 4. Results

### 4.1 Metrics (scaled units)

Metrics are reported in the normalised feature space (StandardScaler units):

| Model | Train RMSE | Train MAE | Val RMSE | Val MAE | Test RMSE | Test MAE |
|---|---|---|---|---|---|---|
| **MLP** | **0.0179** | **0.0133** | **0.0193** | **0.0141** | **0.0181** | **0.0136** |
| DNN | 1.4090 | 0.9449 | 1.0042 | 0.7376 | 1.0759 | 0.7517 |

**Winner: MLP** — test RMSE 0.0181 vs DNN 1.0759.

### 4.2 Prediction Quality (MLP)

Reloading the saved MLP checkpoint and running inference on the first 5 test samples confirms near-perfect agreement with ground truth:

| Sample | Predicted (°C) | Actual (°C) |
|---|---|---|
| 1 | 8.339 | 8.335 |
| 2 | 8.171 | 8.168 |
| 3 | 8.124 | 8.123 |
| 4 | 8.124 | 8.125 |
| 5 | 8.067 | 8.068 |

The maximum absolute error across these five samples is < 0.005 °C, demonstrating that the serialised model reloads cleanly and produces stable predictions.

---

## 5. Analysis

### 5.1 Why MLP Outperforms DNN

The MLP significantly outperforms the deeper DNN on all splits. Several factors explain this outcome:

1. **Lag/rolling features linearise the problem.** The three 1h/2h/3h temperature lags and the 3h/6h/12h rolling means already encode the recent autocorrelation structure of the target. Given that temperature changes slowly (hour-to-hour autocorrelation is very high), the regression is nearly linear in the recent history. A shallow network can fit this relationship as effectively as a deep one.

2. **Dropout over-regularises the DNN.** With Dropout applied after every hidden layer in a network that already has a relatively small number of parameters (14 K), the effective capacity is substantially reduced during training. For a tabular task with a clean, low-noise signal like this, aggressive Dropout acts as an obstacle rather than a safeguard.

3. **Depth provides diminishing returns on tabular data.** Deep architectures are most beneficial when the input contains unstructured or hierarchically compositional patterns (e.g. images, text). For tabular regression where domain-engineered features already capture the relevant structure, adding layers increases training difficulty without improving representational power.

4. **Early stopping favours the simpler model.** The MLP has fewer parameters, converges faster, and its validation loss stabilises at a lower value before early stopping triggers. The DNN may need more epochs or a lower Dropout rate to converge properly.

### 5.2 Generalisation

The MLP shows excellent generalisation: test RMSE (0.0181) is nearly identical to training RMSE (0.0179), indicating no meaningful overfitting. The slight gap between validation RMSE (0.0193) and test RMSE (0.0181) is within noise and does not indicate any issue.

---

## 6. Artifacts

| File | Size | Description |
|---|---|---|
| `output/mlp_best.keras` | 81,086 bytes | Best MLP weights (by val loss) |
| `output/dnn_best.keras` | 222,252 bytes | Best DNN weights (by val loss) |
| `output/scaler.pkl` | 992 bytes | Fitted StandardScaler |

---

## 7. Conclusion

For 1-hour-ahead air temperature forecasting on the Jena Climate dataset, a **shallow MLP with 3 hidden layers (4,225 parameters)** achieves near-zero error (test RMSE ≈ 0.018 in scaled units) and comfortably outperforms a deeper DNN with Dropout. The result underlines that thoughtful feature engineering — specifically lag and rolling mean features that encode autocorrelation — can reduce a time-series regression problem to a near-linear one, where model capacity beyond a simple MLP yields no benefit.
