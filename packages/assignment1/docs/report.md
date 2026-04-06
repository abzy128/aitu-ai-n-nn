# EDA & Feature Engineering Report — Jena Climate Dataset

**Task:** Multivariate time-series regression  
**Target variable:** `T (degC)` (air temperature)  
**Dataset:** Jena Climate 2009–2016

---

## 1. Dataset Overview

The raw dataset contains **420,551 rows × 15 columns**, recorded at 10-minute intervals from January 2009 to January 2017 at the Max Planck Institute for Biogeochemistry weather station in Jena, Germany. All 14 numeric features are recorded without any structural missing values (no NaN in the raw file).

| Property | Value |
|---|---|
| Raw shape | 420,551 × 15 |
| Date range | 2009-01-01 00:10 → 2017-01-01 00:00 |
| Recording interval | 10 minutes |
| Target mean | 9.45 °C |
| Target std | 8.42 °C |
| Target range | −23.0 °C to 37.3 °C |

**Features included:** atmospheric pressure `p (mbar)`, temperature `T (degC)`, potential temperature `Tpot (K)`, dew point `Tdew (degC)`, relative humidity `rh (%)`, saturation vapor pressure `VPmax`, actual vapor pressure `VPact`, vapor pressure deficit `VPdef`, specific humidity `sh (g/kg)`, water vapor concentration `H2OC (mmol/mol)`, air density `rho (g/m³)`, wind velocity `wv (m/s)`, maximum wind velocity `max. wv (m/s)`, wind direction `wd (deg)`.

---

## 2. Timestamp Handling & Resampling

### 2.1 Duplicate Timestamps

Parsing the `Date Time` column to a `DatetimeIndex` revealed **327 duplicate timestamps**. These were resolved by keeping the first occurrence of each duplicate.

### 2.2 Irregular Intervals

The raw data exhibits irregular time deltas beyond the expected 10-minute interval, including gaps of 20 min, 30 min, 16 hours, and up to 3 days. This makes direct use of the high-frequency series impractical for many models.

### 2.3 Hourly Resampling

The 10-minute series was resampled to **hourly resolution** by taking the mean within each hour. This reduces the dataset to **70,129 rows × 14 columns** — a ~6× compression — while smoothing sub-hourly noise and producing a strictly regular index suitable for time-series modelling.

---

## 3. Exploratory Analysis

### 3.1 Correlations with Target

The correlation heatmap reveals strong linear relationships between `T (degC)` and several thermodynamic features:

| Feature | Correlation with T (degC) |
|---|---|
| `Tpot (K)` | ~1.00 (near-perfect, physically expected) |
| `Tdew (degC)` | ~0.99 (dew point tracks air temperature closely) |
| `VPmax (mbar)` | ~0.95 (saturation vapor pressure is exponential in T) |
| `VPact (mbar)` | positive, high |
| `rh (%)` | ~−0.56 (higher T → lower relative humidity at same moisture) |

Features such as `p (mbar)`, `rho (g/m³)`, and wind variables show weaker direct correlation with temperature.

### 3.2 Distribution Analysis

Histograms of all features highlighted two important data quality issues:

- **`wv (m/s)` and `max. wv (m/s)`** contained extreme spikes at **−9999**, indicating sentinel values used to encode missing measurements.
- `wd (deg)` was also flagged as a potential sentinel carrier.
- All other features showed plausible, smooth distributions.

### 3.3 Temperature Time Series

The time series of `T (degC)` displays clear **seasonal periodicity** (annual cycle) and **diurnal cycles** (daily highs and lows), consistent with a mid-latitude continental climate. Rolling means at daily and weekly scales confirm the strong low-frequency structure that lag and rolling features should capture.

### 3.4 Diurnal × Day-of-Week Patterns

A heatmap of mean temperature by hour-of-day (rows) and weekday (columns) shows that the warmest hours are consistently in the early-to-mid afternoon (around 13:00–15:00) regardless of weekday, and no significant weekday effect is present — as expected for a physical climate variable.

---

## 4. Data Cleaning

### 4.1 Sentinel Value Replacement

Sentinel values of `−9999` were detected and replaced with `NaN`:

| Column | Sentinel values replaced |
|---|---|
| `wv (m/s)` | 2 |
| `max. wv (m/s)` | 3 |
| **Total** | **5** |

### 4.2 Missing Value Interpolation

After sentinel replacement, **1,237 NaN values** remained (introduced by resampling gaps and sentinel removal). These were filled using **time-based linear interpolation**, which respects the temporal ordering of the series and avoids introducing discontinuities. After interpolation, **0 missing values** remained.

### 4.3 Outlier Handling

Physical domain knowledge was used to cap values outside plausible measurement ranges rather than dropping them, preserving the temporal continuity of the series:

| Column | Outliers capped | Allowed range |
|---|---|---|
| `wv (m/s)` | 4 | [0.0, 60.0] m/s |
| `max. wv (m/s)` | 4 | [0.0, 80.0] m/s |

Capping (Winsorization) was preferred over row deletion because dropping observations from a time series creates gaps that would need to be re-imputed.

---

## 5. Feature Engineering

Eight new feature columns were derived from the cleaned hourly series, grouped into three categories:

### 5.1 Calendar / Time Features

Extracted from the `DatetimeIndex`:

| Feature | Type | Description |
|---|---|---|
| `hour` | int (0–23) | Hour of day |
| `day_of_week` | int (0–6) | Day of week (Monday=0) |
| `month` | int (1–12) | Calendar month |
| `is_weekend` | binary | 1 if Saturday or Sunday |

These encode known periodic patterns in temperature (diurnal and seasonal cycles).

### 5.2 Lag Features

Previous values of the target variable, providing the model with recent history:

| Feature | Lag |
|---|---|
| `T (degC)_lag_1h` | T at t−1 h |
| `T (degC)_lag_2h` | T at t−2 h |
| `T (degC)_lag_3h` | T at t−3 h |

Given the high autocorrelation in temperature, these are expected to be among the most predictive features.

### 5.3 Rolling Mean Features

Smoothed recent temperature context over multiple time windows:

| Feature | Window |
|---|---|
| `T (degC)_rolling_3` | 3-hour mean |
| `T (degC)_rolling_6` | 6-hour mean |
| `T (degC)_rolling_12` | 12-hour mean |

Rolling averages capture the recent trend and reduce sensitivity to single-hour spikes.

### 5.4 Time-of-Day Category

A categorical variable encoding the period of the day:

| Label | Hours | Ordinal |
|---|---|---|
| night | 00:00–05:59 | 0 |
| morning | 06:00–11:59 | 1 |
| afternoon | 12:00–17:59 | 2 |
| evening | 18:00–23:59 | 3 |

Stored as both a string column (`time_of_day`) and an integer ordinal (`time_of_day_ord`). The four classes are balanced at ~17,532–17,533 rows each.

---

## 6. Edge Effects

The first 3 rows introduced NaN values due to the 3-hour lag and rolling window initialization. These were dropped, leaving a final dataset of **70,126 rows**.

---

## 7. Final Dataset Summary

| Stage | Shape |
|---|---|
| Raw | 420,551 × 15 |
| After hourly resampling | 70,129 × 18 |
| After cleaning | 70,129 × 18 |
| Final (with engineered features) | **70,126 × 26** |

**Final columns (26):**

`p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`, `rh (%)`, `VPmax (mbar)`, `VPact (mbar)`, `VPdef (mbar)`, `sh (g/kg)`, `H2OC (mmol/mol)`, `rho (g/m³)`, `wv (m/s)`, `max. wv (m/s)`, `wd (deg)` — plus engineered: `hour`, `day_of_week`, `month`, `is_weekend`, `T_lag_1h`, `T_lag_2h`, `T_lag_3h`, `T_rolling_3`, `T_rolling_6`, `T_rolling_12`, `time_of_day`, `time_of_day_ord`.

The processed dataset is saved to `output/processed_data.csv` and is ready for model training.

---

## 8. Key Findings

1. **Near-perfect physical proxies exist**: `Tpot (K)` and `Tdew (degC)` are near-perfect correlates of temperature. In a forecasting setting these would typically be unavailable at prediction time, so care must be taken to exclude future-leaking features.
2. **Lag features will dominate**: Temperature is highly autocorrelated; the 1-hour lag alone should explain the majority of variance in a short-horizon forecast.
3. **Data quality issues were minor**: Only 5 sentinel values and 1,237 interpolated values out of ~70,000 hourly observations — the dataset is clean overall.
4. **Strong seasonality**: Annual and diurnal cycles are prominent and well-captured by the calendar features and rolling means.
