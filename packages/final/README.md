# Furnace1 Anomaly Detection

This package trains anomaly detectors for `dataset/Furnace1.csv` and exports
all candidate models for later inference API integration.

Run from this directory:

```bash
uv run python main.py
```

The training script exports:

- `models/isolation_forest.joblib`
- `models/one_class_svm.joblib`
- `models/local_outlier_factor.joblib`
- `models/forecast_residual_hgb.joblib`
- `models/lstm_autoencoder.joblib`
- `models/lstm_autoencoder.pt`

Evaluation reports are written to `reports/metrics.json` and
`reports/metrics.md`.

The dataset has no real anomaly label column, so Precision, Recall and F1 are
computed against proxy labels:

- `active_power < 10.0`
- absolute 1-minute `active_power` change greater than `5.0`

Run the API and chart page:

```bash
uv run uvicorn final_anomaly.api:app --reload
```

Then open <http://127.0.0.1:8000/>.
