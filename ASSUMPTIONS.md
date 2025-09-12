# Energy ML Problem Assumptions

Because the detailed problem statement isn't included in the repository yet, the following scoped assumptions define the initial ML pipeline design. Adjust these once the official spec is available.

## 1. Primary Objective
Short to medium horizon forecasting of aggregated energy demand (kWh) at hourly granularity. Target variable: `load` (float). Forecast horizons: next 24 hours (H=24) given historical data up to "now".

## 2. Data Layout (Expected CSV Schema)
Required columns:
- `timestamp` (UTC ISO8601 or "%Y-%m-%d %H:%M:%S")
- `load` (float) — target
Optional columns (used if present):
- `temperature` (float, °C)
- `humidity` (float, %)
- `wind_speed` (float, m/s)
- `is_holiday` (0/1)
- `price` (float) — energy market price

Assume a single time series (no multiple sites). If later multi-site: add `site_id` and pivot / hierarchical aggregation logic.

## 3. Frequency & Missing Data
- Base frequency: hourly.
- Allow small gaps (< 6 consecutive hours) -> impute via linear interpolation.
- Larger gaps -> forward fill then flag with a binary feature (`gap_flag`).

## 4. Feature Engineering Plan
- Lags: 1, 2, 3, 6, 12, 24, 48 hours.
- Rolling statistics (window: 3, 6, 12, 24, 48): mean, std, min, max.
- Time/calendar: hour_of_day, day_of_week, is_weekend, month, day_of_year.
- Interaction: temperature * humidity (if available).
- Holiday expansion: if `is_holiday` absent, derive from country calendar (configurable, default none).

## 5. Train / Validation / Test Split
- Chronological split.
- Last 14 days -> test.
- Prior 14 days -> validation.
- Remainder -> train.
(Adjust via config.)

## 6. Models (Initial Set)
1. Baseline Naive (persistence: predict last observed load).
2. Linear Regression with regularization (Ridge).
3. Tree ensemble (RandomForestRegressor).
4. Gradient Boosting (HistGradientBoostingRegressor) for speed.
5. (Optional) LSTM (only if PyTorch installed; otherwise skipped gracefully).

Ensembling: simple weighted average (optional; default off).

## 7. Metrics
- MAE (primary)
- RMSE
- MAPE (with safeguard when target near zero)
- R²

## 8. Artifact Outputs
Saved under `artifacts/`:
- `model.pkl`
- `scaler.pkl` (if scaling used)
- `feature_spec.json`
- `metrics.json`
- `config_used.json`

## 9. Configuration
Pydantic `Settings` loaded from (precedence): CLI args > env vars > `.env` file > defaults. Key parameters:
- `data_path`
- `timestamp_col`, `target_col`
- `freq`
- `horizon`
- `validation_days`, `test_days`
- `model_type`
- `random_seed`

## 10. Forecasting Strategy
Direct multi-step regression: train one model to predict t+1 then recursively produce horizon. (Future enhancement: multi-output regression or separate models per horizon.)

## 11. Reproducibility
- Fixed random seeds.
- Deterministic splits recorded in `split_info.json`.

## 12. Risks / Open Points
- If actual task is anomaly detection: need different pipeline (isolation forests, thresholds) — would refactor.
- If multiple sites: switch to panel features (group lags) and possibly global forecasting models.
- If sub-hourly granularity: adjust frequency and lag sets.

## 13. Next Steps
1. Scaffold package structure.
2. Implement config & data loading.
3. Add feature engineering + models.
4. Add training / evaluation scripts.
5. Provide synthetic data generator + quick run instructions.

---
Update this document once the real spec is available to avoid misalignment.
