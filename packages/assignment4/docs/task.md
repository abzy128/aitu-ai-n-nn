# Implementation Plan — 1D CNN Temperature Forecasting
**Dataset:** Weather Dataset (Jena) · **Target:** `T (degC)` · **Task:** Regression (adapted from classification assignment)
**Builds on:** `output/processed_data.csv` (Stage 1), `scaler.pkl` (Stage 2), LSTM metrics (Stage 3)

---

## Adaptation Notes
The original assignment frames this as binary classification (high/low consumption). Since we are doing **temperature forecasting** on Jena Weather, this stage is adapted as follows:

| Original | Adapted |
|---|---|
| Binarize target → `HighConsumption` | Keep `T (degC)` as continuous target |
| `binary_crossentropy` + sigmoid | `mse` loss + linear output |
| Accuracy / Precision / Recall / F1 | RMSE / MAE (°C), comparable to Stages 2 & 3 |
| Compare vs Logistic Regression | Compare vs **Persistence baseline** (standard for time series) |
| Classification confusion matrix | Residual error distribution plot |

Everything else — sliding windows, chronological split, EarlyStopping, model saving — stays the same.

---

## Project Structure

```
zadanie_4/
├── notebooks/
│   └── 04_cnn.ipynb
├── utils/
│   ├── __init__.py
│   ├── data_prep.py      # reuse from Stage 2/3
│   ├── sequencer.py      # reuse from Stage 3
│   ├── models.py         # build_cnn_simple(), build_cnn_deep()
│   ├── trainer.py        # reuse from Stage 3
│   └── evaluator.py      # reuse from Stage 3 + residual plot
├── output/
│   ├── processed_data.csv  # from Stage 1
│   ├── scaler.pkl          # from Stage 2 — reload, do NOT refit
│   └── cnn_best.h5         # best CNN checkpoint
└── requirements.txt
```

---

## Step 1 — Data Preparation & Sequences
**File:** `utils/data_prep.py` + `utils/sequencer.py` · **Notebook:** Section 1

- Load `output/processed_data.csv`, parse datetime index, drop NaN rows
- Reload `scaler.pkl` from Stage 2 — `.transform()` only, including target (same convention as Stage 3)
- Chronological split at identical 70/15/15 boundaries as Stages 2 & 3
- Build sliding windows with `window_size=24` (best from Stage 3 experiment) — reuse `sequencer.py` directly
- Confirm output shapes: `X → (samples, 24, n_features)`, `y → (samples,)`

> No new data prep logic needed — this step is purely reuse from Stage 3.

---

## Step 2 — Persistence Baseline
**Notebook:** Section 2

- **Persistence model:** predict that the next temperature = the last temperature in the window (`y_pred = X[:, -1, target_feature_idx]`), inverse-transformed to °C
- Compute RMSE and MAE on the test set
- This is the standard "naive" benchmark for time series — a model that cannot beat persistence is not useful
- Add to the running cross-stage comparison table

---

## Step 3 — Simple 1D CNN
**File:** `utils/models.py` · **Notebook:** Section 3

Architecture:
```
Input(shape=(24, n_features))
→ Conv1D(64 filters, kernel_size=3, activation='relu', padding='causal')
→ MaxPooling1D(pool_size=2)
→ Conv1D(32 filters, kernel_size=3, activation='relu', padding='causal')
→ GlobalAveragePooling1D()
→ Dense(32, relu)
→ Dense(1, linear)
```

- `padding='causal'` ensures the convolution only looks at past timesteps — no future leakage
- `GlobalAveragePooling1D` instead of `Flatten` — more robust to window size changes and fewer parameters
- Compile: `Adam(lr=1e-3)`, `loss=mse`, `metrics=[mae]`
- Fit with `EarlyStopping(patience=10, restore_best_weights=True)` + `ModelCheckpoint('output/cnn_best.h5')`
- Plot train vs val loss

---

## Step 4 — Deep Multi-Scale CNN
**File:** `utils/models.py` · **Notebook:** Section 4

Architecture — two parallel convolution branches with different kernel sizes, merged before output:
```
Input
├── Conv1D(64, kernel_size=3, causal) → MaxPooling1D
└── Conv1D(64, kernel_size=7, causal) → MaxPooling1D
→ Concatenate
→ Conv1D(32, kernel_size=3, causal)
→ GlobalAveragePooling1D
→ Dense(32, relu) → Dropout(0.2)
→ Dense(1, linear)
```

- **Rationale:** short kernels (3) capture rapid fluctuations; long kernels (7) capture slower daily patterns — both relevant for temperature
- Same compile/fit settings as simple CNN
- Compare training curves of simple vs deep CNN on a single plot

---

## Step 5 — Evaluation & Full Cross-Stage Comparison
**File:** `utils/evaluator.py` · **Notebook:** Section 5

- Inverse-transform all predictions before metrics
- Compute RMSE and MAE for both CNN variants on test set
- **Full comparison table** across all stages:

  | Model             | Test RMSE (°C) | Test MAE (°C) |
  |-------------------|----------------|---------------|
  | Persistence       |                |               |
  | MLP (Stage 2)     |                |               |
  | DNN (Stage 2)     |                |               |
  | LSTM 1L (Stage 3) |                |               |
  | LSTM 2L (Stage 3) |                |               |
  | CNN Simple        |                |               |
  | CNN Multi-Scale   |                |               |

- **Actual vs Predicted plot** — first 500 test samples, same format as Stages 2 & 3
- **Residual distribution plot** — histogram of `(y_pred - y_true)` in °C; should be centered near 0 with narrow spread
- Written conclusion: where does CNN sit vs LSTM? Does it train faster? Is it competitive? When would you prefer one over the other?

---

## Step 6 — Save Artifacts
**Notebook:** Section 6

- `output/cnn_best.h5` saved by `ModelCheckpoint` during best-performing CNN training
- Reload and run one forward pass on test set to verify
- Print final test RMSE/MAE from reloaded model

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

- [ ] `scaler.pkl` reloaded, target scaled, inverse-transform before metrics
- [ ] Sequences reused from Stage 3 (`window_size=24`)
- [ ] Persistence baseline computed and included in comparison table
- [ ] Simple CNN: `Conv1D → MaxPooling1D → Conv1D → GlobalAveragePooling1D → Dense`
- [ ] `padding='causal'` used on all Conv1D layers
- [ ] Multi-scale CNN: two parallel branches with kernel sizes 3 and 7
- [ ] EarlyStopping + ModelCheckpoint used for both CNN variants
- [ ] Full cross-stage comparison table (Persistence → MLP → DNN → LSTM → CNN)
- [ ] Residual distribution plot included
- [ ] `cnn_best.h5` saved and reload-verified
- [ ] Notebook fully re-runnable top to bottom
