# Assignment 6 — Siamese Network for Weather Anomaly Detection

**Dataset:** Jena Climate (2009–2016)  
**Task:** Metric learning for weather anomaly detection using a Siamese neural network

## Overview

Builds on all previous stages, especially Stage 5 (autoencoder) for anomaly labels and Stage 4 (CNN encoder) for the shared subnet architecture.

| Stage | Description |
|-------|-------------|
| 5 | Autoencoder reconstruction error → binary anomaly labels (p95 threshold) |
| 4 | CNN simple architecture → stripped of Dense(1) head → shared embedding subnet |
| 6 | Siamese network with contrastive loss → metric-based anomaly detection |

## Architecture

```
Input_A (24, n_features) ─┐
                           ├─→ SharedCNN → embedding_A ─┐
Input_B (24, n_features) ─┘                              ├─→ Euclidean distance → output
                           SharedCNN → embedding_B ──────┘
```

**SharedCNN** (identical weights in both branches):
- Conv1D(64, k=3, causal) → MaxPool1D(2)
- Conv1D(32, k=3, causal) → GlobalAveragePooling1D
- Dense(32, relu) → Dense(128, relu) → L2Normalize

**Contrastive Loss:**
```
L = (1-y) * 0.5 * D²  +  y * 0.5 * max(0, margin - D)²
```
- `y=0` (similar pair): distance should be small  
- `y=1` (dissimilar pair): distance should exceed margin (1.0)

## Structure

```
assignment6/
├── notebooks/
│   └── 06_siamese.ipynb      # main notebook, all 9 sections
├── utils/
│   ├── data_prep.py           # load, scale, split (reused from Stage 5)
│   ├── sequencer.py           # sliding-window builder (reused from Stage 5)
│   ├── pairs.py               # anomaly labeling + pair construction
│   ├── models.py              # build_shared_cnn(), build_siamese()
│   ├── losses.py              # contrastive_loss()
│   └── evaluator.py           # calibration, ROC, kNN scoring, plots
├── output/
│   ├── processed_data.csv     # → Stage 1 (symlink)
│   ├── scaler.pkl             # → Stage 2 (symlink)
│   ├── autoencoder.h5         # → Stage 5 (symlink)
│   ├── siamese.h5             # trained Siamese model (generated)
│   └── shared_cnn.h5          # shared subnet (generated)
└── reports/
    └── metrics_summary.csv    # cross-stage comparison table (generated)
```

## Running

```bash
cd notebooks/
jupyter nbconvert --to notebook --execute 06_siamese.ipynb --output 06_siamese_executed.ipynb
```

Or open interactively:
```bash
jupyter lab notebooks/06_siamese.ipynb
```

## Results

After training, the notebook produces:
- `output/siamese.h5` and `output/shared_cnn.h5`
- ROC curve with AUC score
- Distance distribution plot (positive vs negative pairs)
- Nearest-neighbor visualization (anomalous test windows vs closest normal)
- Cross-stage metrics summary CSV
