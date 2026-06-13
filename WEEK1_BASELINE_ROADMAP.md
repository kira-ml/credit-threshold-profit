# Week 1 Roadmap — Baseline Version

**Goal:** End the week with a working, end-to-end pipeline that loads data, computes NPV, trains simple models, evaluates profit uplift, and saves results. No tuning. No feature engineering beyond raw columns. Everything runs start to finish.

---

## Day 1 — Data Ingestion

- [ ] Download Lending Club dataset from Kaggle
- [ ] Convert CSV to Parquet using Polars
- [ ] Load Parquet, inspect shape, column names, dtypes
- [ ] Filter to loans that have reached final status (fully paid or charged off)
- [ ] Save filtered Parquet to `data/processed/`

**Deliverable:** `data/processed/loans_filtered.parquet`

---

## Day 2 — Target Variable

- [ ] Define discount rate (e.g., 2%)
- [ ] Compute per-loan NPV from: total payments received, recoveries, charge-off amounts, loan amount, term, interest rate
- [ ] Validate distribution of NPV — check min, max, mean, share of negative values
- [ ] Save DataFrame with target column added

**Deliverable:** `data/processed/loans_with_npv.parquet`

---

## Day 3 — EDA and Train/Test Split

- [ ] Notebook: summary stats, distributions, missingness, correlations with NPV
- [ ] Identify and drop columns not available at origination (post-issue payment fields, outcome flags)
- [ ] Sort by issue date, split 60/20/20
- [ ] Save train/val/test Parquet files

**Deliverable:** `data/processed/train.parquet`, `val.parquet`, `test.parquet`

---

## Day 4 — Baselines

- [ ] Select numeric + low-cardinality categorical features available at origination
- [ ] Mean predictor: compute and store train mean NPV
- [ ] Linear regression: fit on train, predict on val and test
- [ ] Shallow decision tree (max_depth=5): fit on train, predict on val and test
- [ ] Save predictions for each model

**Deliverable:** `outputs/predictions_baselines.csv` (model name, split, loan index, predicted NPV)

---

## Day 5 — Evaluation (Model Metrics)

- [ ] Compute RMSE, MAE, R², Spearman rank correlation for each model on test set
- [ ] Compare all three baselines in a single table
- [ ] Quick residual plots for linear regression

**Deliverable:** `outputs/model_metrics.csv`

---

## Day 6 — Profit Simulation

- [ ] Implement profit-threshold simulator: for each model, sweep thresholds, compute aggregate portfolio profit on test set
- [ ] Add approve-all baseline (θ = -∞)
- [ ] Plot profit vs. acceptance rate curve for each model
- [ ] Identify best threshold per model on validation set, apply to test set

**Deliverable:** `outputs/profit_simulation.csv`, `outputs/profit_curve.png`

---

## Day 7 — Wrap Up

- [ ] Write a one-paragraph summary of which model performed best by profit uplift
- [ ] Ensure all scripts run sequentially without errors
- [ ] Clean notebook(s), add minimal markdown headers
- [ ] Commit everything, push to `initial-build`

**Deliverable:** Repo runs end to end. Ready to plan Week 2.