# Implementation Plan — EDA & Feature Engineering
**Dataset:** Weather Dataset (Jena) · **Task:** Regression, multivariate time series

---

## Project Structure

```
zadanie_1/
├── data/
│   └── jena_climate_2009_2016.csv
├── notebooks/
│   └── 01_eda.ipynb
├── utils/
│   ├── __init__.py
│   ├── loader.py        # data loading & validation
│   ├── time_features.py # timestamp parsing & extraction
│   ├── visualizer.py    # all plotting functions
│   ├── cleaner.py       # missing values & outlier handling
│   └── features.py      # feature engineering (lags, rolling, categories)
├── output/
│   └── processed_data.csv
└── requirements.txt
```

---

## Step 1 — Data Loading & Initial Inspection
**File:** `utils/loader.py` · **Notebook:** Section 1

- Download dataset from https://www.bgc-jena.mpg.de/wetter/
- Load CSV with `pandas`, detect encoding and separator (`;` in Jena dataset)
- Print shape, dtypes, `.info()`, `.describe()`
- Check for missing values per column

> **Jena-specific note:** the dataset uses `;` as separator and has a `Date Time` column in `dd.mm.yyyy HH:MM:SS` format. Target variable is `T (degC)` (air temperature).

---

## Step 2 — Timestamp Handling
**File:** `utils/time_features.py` · **Notebook:** Section 2

- Parse `Date Time` column into `datetime` and set as index
- Extract: `hour`, `day_of_week`, `month`, `is_weekend`
- Verify chronological order; assert no gaps or duplicates
- Resample to hourly if needed (raw data is every 10 min)

---

## Step 3 — Visualization & Distribution Analysis
**File:** `utils/visualizer.py` · **Notebook:** Section 3

- **Correlation matrix** — heatmap of all numeric features vs `T (degC)`; note top correlated features (`Tdew`, `rh`, `p`)
- **Histograms** — distribution of each feature; flag suspicious spikes (e.g. `wv` has known `-9999` sentinel values)
- **Time series plot** — raw `T (degC)` + rolling mean (24h, 7d windows) to show daily/seasonal cycles
- **Consumption by hour & weekday** — mean temperature heatmap (hour × day_of_week)

---

## Step 4 — Cleaning: Missing Values & Outliers
**File:** `utils/cleaner.py` · **Notebook:** Section 4

- Replace known sentinel values (`-9999` in wind speed/direction) with `NaN`
- Fill remaining NaNs with time-based interpolation (`method='time'`)
- Detect outliers via IQR or z-score per column; cap or drop based on physical plausibility (e.g. `T` outside −30…+45°C is suspect)
- Document every decision with a comment in the notebook

---

## Step 5 — Feature Engineering
**File:** `utils/features.py` · **Notebook:** Section 5

- **Lag features** on `T (degC)`: `lag_1h`, `lag_2h`, `lag_3h`
- **Rolling mean** features: 3, 6, 12 observations (30 / 60 / 120 min windows)
- **Time-of-day category**: `night` (0–5), `morning` (6–11), `afternoon` (12–17), `evening` (18–23) — encoded as string or ordinal int
- Drop rows with NaN introduced by lags (first N rows)

---

## Step 6 — Save & Report
**Notebook:** Section 6

- Save cleaned + enriched DataFrame to `output/processed_data.csv`
- Final notebook cell: brief bullet-point summary of findings (shape before/after, key correlations, outlier counts, new column list)

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Evaluation Checklist

- [ ] All 6 task sections implemented
- [ ] Timestamp parsed and new time features extracted
- [ ] Sentinel values (`-9999`) identified and replaced
- [ ] Outliers handled with documented rationale
- [ ] Lag + rolling + time-of-day features present in output CSV
- [ ] Notebook is fully re-runnable top to bottom
- [ ] `processed_data.csv` saved to `output/`

