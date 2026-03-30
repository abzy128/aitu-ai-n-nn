# Implementation Plan — LSTM Regression
**Dataset:** Weather Dataset (Jena) · **Target:** `T (degC)` · **Builds on:** `output/processed_data.csv` from Stage 1, metrics from Stage 2

---

## Project Structure

```
zadanie_3/
├── notebooks/
│   └── 03_lstm.ipynb
├── utils/
│   ├── __init__.py
│   ├── data_prep.py        # reuse/adapt from Stage 2 (scaling + chronological split)
│   ├── sequencer.py        # sliding window → (samples, timesteps, features) arrays
│   ├── models.py           # build_lstm() factory function
│   ├── trainer.py          # compile, fit, EarlyStopping (reuse from Stage 2)
│   └── evaluator.py        # metrics + plots (reuse from Stage 2)
├── output/
│   ├── processed_data.csv  # from Stage 1
│   ├── scaler.pkl          # from Stage 2 — reload, do NOT refit
│   └── lstm_best.h5        # best LSTM checkpoint
└── requirements.txt
```

---

## Step 1 — Data Preparation & Sequence Construction
**File:** `utils/data_prep.py` + `utils/sequencer.py` · **Notebook:** Section 1

- Load `output/processed_data.csv`, parse datetime index, drop NaN rows from lag features
- **Reload** `scaler.pkl` from Stage 2 and call `.transform()` only — never refit on new data to avoid leakage
- **This time, scale the target too** — LSTM training is more stable when all values are in the same range; inverse-transform predictions before computing metrics so RMSE/MAE stays in °C
- Chronological split: same 70/15/15 boundaries as Stage 2 (match by index position, not date, to ensure identical splits)
- **Sliding window** in `sequencer.py`:
  - Function signature: `make_sequences(data, target_col, window_size) → X, y`
  - For each position `i`: `X[i] = data[i : i+window_size]`, `y[i] = target[i+window_size]`
  - Output shapes: `X → (samples, window_size, n_features)`, `y → (samples,)`
  - Apply separately to train/val/test splits — **no cross-boundary sequences**

> **Jena note:** raw data is 10-minute intervals. Window of 24 = 4 hours, 48 = 8 hours, 144 = 24 hours. The assignment suggests 24 as default — good starting point.

---

## Step 2 — LSTM Model
**File:** `utils/models.py` · **Notebook:** Section 2

- **Single-layer LSTM** as baseline:
  `Input → LSTM(64, return_sequences=False) → Dropout(0.2) → Dense(1, linear)`
- **Two-layer LSTM** as main model:
  `Input → LSTM(100, return_sequences=True) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1, linear)`
- Compile: `optimizer=Adam(lr=1e-3)`, `loss=mse`, `metrics=[mae]`
- Fit with:
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `ModelCheckpoint('output/lstm_best.h5', save_best_only=True)`
  - `batch_size=64` (larger batches help LSTM training stability)
- Plot train vs val loss curve

> **`return_sequences=True`** is required on the first LSTM layer when stacking — it passes the full hidden state sequence to the next LSTM layer instead of just the last step.

---

## Step 3 — Evaluation & Comparison with Stage 2
**File:** `utils/evaluator.py` · **Notebook:** Section 3

- Inverse-transform predictions and ground truth before metric calculation
- Compute RMSE and MAE on test set
- **Cross-stage comparison table** (pull Stage 2 numbers from saved results):

  | Model      | Test RMSE (°C) | Test MAE (°C) |
  |------------|----------------|---------------|
  | MLP        |                |               |
  | DNN        |                |               |
  | LSTM (1L)  |                |               |
  | LSTM (2L)  |                |               |

- **Actual vs Predicted plot** — line plot of first 500 test predictions overlaid on ground truth (same format as Stage 2 for easy visual comparison)
- Written conclusion: does LSTM improve over MLP/DNN, and why (temporal dependencies captured by recurrence vs flat feature vectors)

---

## Step 4 — Window Size Experiment
**Notebook:** Section 4

- Train the best LSTM architecture with window sizes `[12, 24, 48]` (keep all other hyperparameters fixed)
- Record test RMSE and MAE for each window size
- Present as a small results table + line plot (window size on x-axis, RMSE on y-axis)
- Conclusion: diminishing returns past a certain window, or clear optimum?

> **Tip:** wrap training in a loop over window sizes and store results in a dict — avoids code duplication and makes the experiment reproducible in one cell.

---

## Step 5 — Save Artifacts
**Notebook:** Section 5

- `output/lstm_best.h5` saved by `ModelCheckpoint` during training
- Reload model, run one forward pass on test set, assert output shape matches `y_test`
- Print final test RMSE/MAE from reloaded model as the "official" reported result

---

## Requirements

```
tensorflow
keras
pandas
numpy
matplotlib
scikit-learn
jupyter
```

---

## Evaluation Checklist

- [ ] `scaler.pkl` reloaded from Stage 2, not refit
- [ ] Target scaled for training, inverse-transformed for metrics
- [ ] Sequences constructed with no cross-boundary leakage (train/val/test split before windowing)
- [ ] Output shapes confirmed: `X → (samples, timesteps, features)`, `y → (samples,)`
- [ ] Single-layer and two-layer LSTM both implemented
- [ ] `return_sequences=True` on first LSTM when stacking
- [ ] EarlyStopping + ModelCheckpoint used
- [ ] Cross-stage comparison table (MLP / DNN / LSTM) present
- [ ] Window size experiment: 12, 24, 48 tested and compared
- [ ] `lstm_best.h5` saved and reload-verified
- [ ] Notebook fully re-runnable top to bottom
