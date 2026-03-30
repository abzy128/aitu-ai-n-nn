# Implementation Plan — MLP & DNN Regression
**Dataset:** Weather Dataset (Jena) · **Target:** `T (degC)` · **Builds on:** `output/processed_data.csv` from Stage 1

---

## Project Structure

```
zadanie_2/
├── notebooks/
│   └── 02_mlp_dnn.ipynb
├── utils/
│   ├── __init__.py
│   ├── data_prep.py     # loading, scaling, chronological splitting
│   ├── models.py        # build_mlp() and build_dnn() factory functions
│   ├── trainer.py       # compile, fit, EarlyStopping logic
│   └── evaluator.py     # metrics calculation and plotting
├── output/
│   ├── processed_data.csv   # from Stage 1
│   ├── scaler.pkl           # saved StandardScaler
│   └── mlp_best.h5          # best model checkpoint
└── requirements.txt
```

---

## Step 1 — Data Preparation
**File:** `utils/data_prep.py` · **Notebook:** Section 1

- Load `output/processed_data.csv`; parse datetime index
- Define feature set: all numeric columns **except** `T (degC)`, raw timestamp columns, and any leaky columns (e.g. `Tdew` is highly correlated — decide whether to keep)
- Apply `StandardScaler` to features only (target stays unscaled for interpretable RMSE in °C)
- Save fitted scaler to `output/scaler.pkl` for reuse in later stages
- **Chronological split — no shuffle:**
  - Train: first 70%
  - Val: next 15%
  - Test: final 15%
- Print split sizes and date ranges to verify no overlap

> **Jena note:** lag and rolling features from Stage 1 introduce NaNs in the first few rows — drop these before splitting, not after.

---

## Step 2 — MLP Model
**File:** `utils/models.py` · **Notebook:** Section 2

- Architecture: `Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, linear)`
- Compile: `optimizer=Adam(lr=1e-3)`, `loss=mse`, `metrics=[mae]`
- Fit with:
  - `validation_data=(X_val, y_val)`
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `ModelCheckpoint('output/mlp_best.h5', save_best_only=True)`
- Plot training vs validation loss curve

---

## Step 3 — DNN Model
**File:** `utils/models.py` · **Notebook:** Section 3

- Architecture: `Input → Dense(128) → Dense(64) → Dense(32) → Dense(16) → Dense(8) → Dense(1, linear)`, all hidden with ReLU
- Add `Dropout(0.2)` after each hidden layer to counter overfitting from depth
- Same compile/fit/EarlyStopping settings as MLP for fair comparison
- Overlay training curves of MLP vs DNN on a single plot to compare convergence speed and stability

---

## Step 4 — Evaluation
**File:** `utils/evaluator.py` · **Notebook:** Section 4

- Compute **RMSE** and **MAE** (in °C) on train / val / test for both models
- Present as a comparison table:

  | Model | Train RMSE | Val RMSE | Test RMSE | Train MAE | Val MAE | Test MAE |
  |-------|-----------|----------|-----------|-----------|---------|----------|
  | MLP   |           |          |           |           |         |          |
  | DNN   |           |          |           |           |         |          |

- **Actual vs Predicted plot** for test set: line plot of first 500 predictions overlaid on ground truth (easier to read than full test scatter)
- Written conclusion cell: which model wins and why, does depth help for this time series structure

---

## Step 5 — Save Artifacts
**Notebook:** Section 5

- Confirm `output/mlp_best.h5` was saved by `ModelCheckpoint`
- Save the better-performing model explicitly if DNN wins (rename or save separately)
- Reload model and run one forward pass on test set to verify the saved file works

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

- [ ] Features scaled, target left unscaled
- [ ] Split is strictly chronological (no shuffle)
- [ ] Split date ranges printed and verified
- [ ] MLP: 2–3 hidden layers, ReLU, linear output
- [ ] DNN: 5+ hidden layers, Dropout added
- [ ] EarlyStopping used for both models
- [ ] RMSE & MAE reported for all three splits
- [ ] Comparison table present in notebook
- [ ] Actual vs Predicted plot included
- [ ] `mlp_best.h5` saved and reload-verified
- [ ] Notebook fully re-runnable top to bottom
