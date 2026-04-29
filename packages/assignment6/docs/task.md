# Implementation Plan — Siamese Network Anomaly Detection + Final Report
**Dataset:** Weather Dataset (Jena) · **Task:** Metric learning for weather anomaly detection
**Builds on:** All previous stages, especially Stage 5 anomaly flags and Stage 4 CNN encoder

---

## Adaptation Notes
The original assignment forms pairs using the `HighConsumption` binary label. Since our pipeline has no such label, anomaly pairs are derived from Stage 5 autoencoder reconstruction error — which is a stronger, data-driven definition of "anomalous."

| Original | Adapted |
|---|---|
| Positive pairs: `HighConsumption=0` windows | Positive pairs: windows with **low reconstruction error** (normal weather) |
| Negative pairs: normal + anomalous consumption window | Negative pairs: normal window + **high reconstruction error** window (Stage 5 flags) |
| Shared subnet: CNN from Stage 4 (no classifier head) | Shared subnet: CNN encoder from Stage 4 stripped of `Dense(1)` output |
| Anomaly = unusual energy spike | Anomaly = weather pattern the autoencoder found hard to reconstruct |

Everything else — contrastive loss, distance threshold, ROC evaluation, final report — applies directly.

---

## Project Structure

```
zadanie_6/
├── notebooks/
│   └── 06_siamese.ipynb
├── utils/
│   ├── __init__.py
│   ├── data_prep.py       # reuse from Stage 2/3
│   ├── sequencer.py       # reuse from Stage 3
│   ├── pairs.py           # pair generation logic
│   ├── models.py          # build_shared_cnn(), build_siamese()
│   ├── losses.py          # contrastive_loss()
│   └── evaluator.py       # distance threshold, ROC, nearest-neighbor search
├── output/
│   ├── processed_data.csv # from Stage 1
│   ├── scaler.pkl         # from Stage 2
│   ├── autoencoder.h5     # from Stage 5 — to derive anomaly labels
│   ├── siamese.h5         # full siamese model
│   └── shared_cnn.h5      # shared subnet only (feature extractor)
└── requirements.txt

── Final deliverables ────────────────────────────────
├── reports/
│   └── final_report.pdf
├── presentations/
│   └── slides.pptx
└── README.md
```

---

## Step 1 — Derive Anomaly Labels from Stage 5
**File:** `utils/pairs.py` · **Notebook:** Section 1

- Load sequences and `scaler.pkl`, build windows with `window_size=24` (same as all prior stages)
- Reload `output/autoencoder.h5` from Stage 5
- Compute per-window reconstruction MSE on the full dataset
- Set anomaly threshold = 95th percentile of **train** reconstruction errors (same rule as Stage 5)
- Assign binary label: `anomaly=1` if MSE > threshold, else `anomaly=0`
- Report label distribution: confirm both classes have sufficient samples (expect ~5% anomalies)
- **Visualize:** plot reconstruction error distribution with threshold line marked

> This reuses Stage 5 work directly — no new labeling logic needed, just loading and applying the saved model.

---

## Step 2 — Construct Training Pairs
**File:** `utils/pairs.py` · **Notebook:** Section 2

- **Positive pairs** (label=0, "same class"): two windows both with `anomaly=0` — similar normal weather patterns
- **Negative pairs** (label=1, "different class"): one window with `anomaly=0` paired with one with `anomaly=1`
- Sample strategy:
  - For each anomalous window, randomly sample 3 normal windows as negative partners
  - For positive pairs, sample an equal number of random normal-normal pairs
  - Result: balanced dataset of ~50% positive, ~50% negative pairs
- Split pairs chronologically: pairs where **both** windows fall in train/val/test respectively
- Print pair counts per split

> Pair construction is the most error-prone step. The key invariant: no window from the test set appears in training pairs — even as the "anchor" window.

---

## Step 3 — Shared CNN Subnet
**File:** `utils/models.py` · **Notebook:** Section 3

- Take the Stage 4 simple CNN architecture, remove the final `Dense(1, linear)` output layer
- Add a `Dense(128, relu)` embedding layer after `GlobalAveragePooling1D` — this is the feature vector
- Add `Lambda(lambda x: tf.math.l2_normalize(x, axis=1))` to normalize embeddings to unit sphere
- **This subnet is shared** — both branches of the Siamese network use identical weights (same Keras model object, not two copies)
- Save standalone as `output/shared_cnn.h5` after training

---

## Step 4 — Siamese Network & Contrastive Loss
**File:** `utils/models.py` + `utils/losses.py` · **Notebook:** Section 4

Full model:
```
Input_A (24, n_features) ─┐
                           ├─→ shared_cnn → embedding_A ─┐
Input_B (24, n_features) ─┘                               ├─→ euclidean_distance → output
                           └─→ shared_cnn → embedding_B ─┘
```

- Euclidean distance layer as a `Lambda` layer:
  `distance = sqrt(sum((embedding_A - embedding_B)^2, axis=1))`

Contrastive loss in `utils/losses.py`:
```
L = (1 - y) * 0.5 * D²
  + y       * 0.5 * max(0, margin - D)²
```
where `y=0` for similar pairs (should be close), `y=1` for dissimilar (should be far), `margin=1.0`

- Compile: `Adam(lr=1e-4)`, custom `contrastive_loss`
- Track `binary_accuracy` with a threshold-based metric (distance < 0.5 → same class)
- Fit with `EarlyStopping(patience=10, restore_best_weights=True)`
- Plot training loss and accuracy curves

---

## Step 5 — Distance Threshold Calibration
**Notebook:** Section 5

- Run the trained Siamese network on all validation pairs
- Plot distribution of distances for positive pairs (normal-normal) vs negative pairs (normal-anomaly)
- Choose threshold `τ` where the two distributions separate best — use F1-score maximization over a grid of candidate thresholds
- Report chosen `τ` and the validation precision/recall at that threshold

---

## Step 6 — Anomaly Detection on Test Set
**File:** `utils/evaluator.py` · **Notebook:** Section 6

- **Nearest-neighbor anomaly score:** for each test window, compute its distance to the 10 nearest normal windows from the training set (using the shared CNN embedding + brute-force cosine distance via `sklearn.metrics.pairwise`)
- Use mean distance to k-nearest normals as the anomaly score
- Flag windows where this score exceeds `τ` as anomalous
- **ROC curve:** use reconstruction error from Stage 5 as ground truth labels; plot ROC and compute AUC
- **Visualization:** pick 3 anomalous test windows and their nearest normal neighbor; plot `T (degC)` for both side-by-side — do they look visually different?
- Report: AUC, precision, recall at chosen threshold

---

## Step 7 — Full Cross-Stage Comparison Table
**Notebook:** Section 7

Consolidate metrics from all stages into a single summary:

| Stage | Model | Task | Primary Metric | Score |
|-------|-------|------|----------------|-------|
| 2 | MLP | Regression | Test RMSE (°C) | |
| 2 | DNN | Regression | Test RMSE (°C) | |
| 3 | LSTM 1L | Regression | Test RMSE (°C) | |
| 3 | LSTM 2L | Regression | Test RMSE (°C) | |
| 4 | Persistence | Regression (baseline) | Test RMSE (°C) | |
| 4 | CNN Simple | Regression | Test RMSE (°C) | |
| 4 | CNN Multi-Scale | Regression | Test RMSE (°C) | |
| 5 | Autoencoder | Anomaly Detection | Anomaly Rate (%) | |
| 6 | Siamese Network | Anomaly Detection | AUC / F1 | |

Written analysis (one paragraph each):
- **Best for forecasting:** which model and why (likely LSTM or CNN — temporal structure)
- **Best for anomaly detection:** Autoencoder vs Siamese — different strengths (reconstruction-based vs metric-based)
- **Architectural takeaways:** what each inductive bias (recurrence vs convolution vs metric learning) is suited to

---

## Step 8 — Save Artifacts
**Notebook:** Section 8

- `output/siamese.h5` — full Siamese model
- `output/shared_cnn.h5` — shared subnet (general-purpose weather pattern embedding)
- Reload both, verify forward passes
- Export final metric table as `reports/metrics_summary.csv`

---

## Step 9 — Final Report & Repository
**Deliverables:** `reports/final_report.pdf` + `README.md` + `slides.pptx`

### Report Structure (scientific article format)
1. **Introduction** — weather forecasting problem, why deep learning, dataset overview
2. **Related Work** — brief review of MLP/LSTM/CNN for time series; autoencoders for anomaly detection; Siamese networks for metric learning
3. **Methodology** — one subsection per model family; data pipeline; feature engineering
4. **Experiments** — training setup, hyperparameters, hardware; all metrics from Step 7 table
5. **Results & Discussion** — cross-model comparison; which architecture won and why; failure cases
6. **Conclusion** — summary of findings; limitations; future directions (e.g. Transformer-based models, larger datasets)
7. **References** — cite original dataset, Keras docs, key papers (LeCun 1989 for CNN, Hochreiter 1997 for LSTM, Bromley 1993 for Siamese)

### Repository Structure
```
/
├── data/                  # raw + processed CSV (gitignore raw if large)
├── notebooks/             # 01_eda.ipynb … 06_siamese.ipynb
├── src/
│   └── utils/             # all shared utility modules
├── models/                # all saved .h5 files
├── reports/
│   ├── final_report.pdf
│   └── metrics_summary.csv
├── presentations/
│   └── slides.pptx
├── requirements.txt
└── README.md              # setup instructions, project summary, results table
```

### Presentation Outline (7–10 min)
1. Problem & dataset (1 min)
2. Data pipeline — EDA, features, windowing (1 min)
3. Forecasting models: MLP → DNN → LSTM → CNN, with comparison chart (2 min)
4. Anomaly detection: Autoencoder reconstruction error + Siamese distance (2 min)
5. Final comparison table + key findings (1 min)
6. Demo / live notebook run (optional, 1–2 min)

---

## Requirements

```
tensorflow
keras
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
fpdf2 or reportlab  # for PDF report generation
python-pptx         # for slides
```

---

## Evaluation Checklist

- [ ] Anomaly labels derived from Stage 5 reconstruction error, not hand-crafted
- [ ] Pair construction: no test windows leak into training pairs
- [ ] Shared CNN subnet: Stage 4 architecture minus final Dense(1), plus Dense(128) embedding
- [ ] L2 normalization on embeddings
- [ ] Contrastive loss implemented manually (not from a library)
- [ ] Distance threshold calibrated on validation set via F1 maximization
- [ ] ROC curve + AUC reported for test anomaly detection
- [ ] Nearest-neighbor visualization: anomalous vs closest normal window plotted
- [ ] Full cross-stage comparison table (Stages 2–6)
- [ ] `siamese.h5` and `shared_cnn.h5` saved and reload-verified
- [ ] GitHub repo with all required folders and README
- [ ] Final report in scientific article format (7 sections)
- [ ] Presentation prepared (7–10 min outline)
- [ ] Notebook fully re-runnable top to bottom
