# Furnace1 Anomaly Detection Metrics

Dataset: `/home/abzy/dev/aitu/masters/trimester3/aitu-ai-n-nn/packages/final/dataset/Furnace1.csv`

## Splits

| Split | Rows | Proxy anomalies |
|---|---:|---:|
| Train | 4854 | 94 |
| Validation | 1618 | 3 |
| Test | 1618 | 72 |

## Models

| Rank | Model | Precision | Recall | F1 | Threshold | Artifact |
|---:|---|---:|---:|---:|---:|---|
| 1 | `forecast_residual_hgb` | 0.7889 | 0.9861 | 0.8765 | 1.228553 | `/home/abzy/dev/aitu/masters/trimester3/aitu-ai-n-nn/packages/final/models/forecast_residual_hgb.joblib` |
| 2 | `one_class_svm` | 0.6667 | 1.0000 | 0.8000 | 2.524655 | `/home/abzy/dev/aitu/masters/trimester3/aitu-ai-n-nn/packages/final/models/one_class_svm.joblib` |
| 3 | `isolation_forest` | 0.6600 | 0.9167 | 0.7674 | 0.148708 | `/home/abzy/dev/aitu/masters/trimester3/aitu-ai-n-nn/packages/final/models/isolation_forest.joblib` |
| 4 | `lstm_autoencoder` | 0.2136 | 0.8750 | 0.3433 | 2.567798 | `/home/abzy/dev/aitu/masters/trimester3/aitu-ai-n-nn/packages/final/models/lstm_autoencoder.joblib` |
| 5 | `local_outlier_factor` | 1.0000 | 0.0556 | 0.1053 | 16.834864 | `/home/abzy/dev/aitu/masters/trimester3/aitu-ai-n-nn/packages/final/models/local_outlier_factor.joblib` |

Metrics use proxy labels: active_power below 10.0 or absolute one-minute change above 5.0.
