# TODO.md

## Credit Threshold Profit Project

*Last Updated: June 16, 2026, 1:45 AM*

---

## ✅ Completed (as of 1:45 AM)

| Category | Task | Status |
|----------|------|--------|
| **Data Pipeline** | CSV to Parquet converter | ✅ Done |
| **EDA** | Comprehensive EDA report + 6 visualizations | ✅ Done |
| **Data Cleaning** | Baseline cleaning (1.34M rows, 110 cols) | ✅ Done |
| **Profit Calculation** | Baseline method working, profit = $555M total | ✅ Done |
| **Preprocessing** | Sparse columns dropped, categories converted, `grade_num` + `emp_length_num` created | ✅ Done |
| **Data Splitting** | 80/20 train/test split by `issue_d` | ✅ Done |
| **Feature Engineering** | All 4 plans generated (`baseline_minimal`, `domain_enhanced`, `interaction_heavy`, `ml_informed`) | ✅ Done |
| **Git History Cleanup** | Removed large Parquet file from Git history | ✅ Done |
| **Git Remote** | Re-added remote and pushed clean branches | ✅ Done |
| **Data Validation** | Complete validation report generated | ✅ Done |
| **Data Quality Fixes** | Outlier capping, moderate missing value imputation added to preprocessing | ✅ Done |

---

## 🚧 Current State (as of 1:45 AM)

| Component | Status | Notes |
|-----------|--------|-------|
| **Profit Column** | ✅ Fixed | $555M total profit, present in `cleaned_baseline.parquet` |
| **Data Pipeline** | ✅ Complete | Cleaning → Profit → Preprocessing → Split → Feature Engineering |
| **Model Training Script** | ⚠️ Needs re-run | Previous run had zero profit. Now profit is fixed. |
| **Logistic Regression** | ⚠️ Needs re-run | Will train on all 4 plans with proper profit |
| **Random Forest** | ⚠️ Needs re-run | Will train on all 4 plans with proper profit |
| **Profit Curves** | ⚠️ Needs re-run | Currently all zeros due to missing profit |
| **Results DataFrame** | ⚠️ Needs re-run | Will generate after successful training |

---

## 🔴 Priority Tasks (Tomorrow - June 16)

| # | Task | Priority | Location | Est. Time |
|---|------|----------|----------|-----------|
| 1 | **Re-run entire pipeline with fixed profit** | 🔴 Critical | CLI | 10 min |
|    | └─ `python src/data_preprocessing.py` | | | |
|    | └─ `python src/data_splitter.py` | | | |
|    | └─ `python src/feature_engineering.py` | | | |
| 2 | **Re-run model training** | 🔴 Critical | CLI | 30-60 min |
|    | └─ `python src/model_training.py` | | | |
| 3 | **Validate profit curves** (should be non-zero) | 🔴 Critical | `reports/model_comparison/` | 5 min |
| 4 | **Generate final results summary** | 🔴 Critical | `model_training.py` | 5 min |
| 5 | **Push `feature/complete-model-training` branch** | 🔴 Critical | CLI | 2 min |

---

## 🟡 Secondary Tasks (After Training Works)

| # | Task | Priority | Location | Est. Time |
|---|------|----------|----------|-----------|
| 1 | Compare AUC, Brier, optimal threshold across all 8 model/plan combos | 🟠 Medium | `model_training.py` | 10 min |
| 2 | Identify best model + feature plan | 🟠 Medium | Analysis | 10 min |
| 3 | Bootstrap confidence intervals for optimal profit | 🟠 Medium | New script | 30 min |
| 4 | Stress test: shift default probability by ±20% | 🟠 Medium | New script | 30 min |

---

## 🔵 Final Deliverables (After All Training Complete)

| # | Task | Priority | Est. Time |
|---|------|----------|-----------|
| 1 | Create final profit curve visualization (publication quality) | 🔴 High | 20 min |
| 2 | Write executive summary (1 paragraph, business impact in $) | 🔴 High | 20 min |
| 3 | Update README.md with results and profit curve | 🟠 Medium | 20 min |
| 4 | Push final version to GitHub | 🟠 Medium | 10 min |

---

## 📊 Expected Metrics (After Re-run)

| Feature Plan | AUC (LR) | AUC (RF) | Optimal Threshold | Optimal Profit |
|--------------|----------|----------|------------------|----------------|
| baseline_minimal | *Pending* | *Pending* | *Pending* | *Pending* |
| domain_enhanced | *Pending* | *Pending* | *Pending* | *Pending* |
| interaction_heavy | *Pending* | *Pending* | *Pending* | *Pending* |
| ml_informed | *Pending* | *Pending* | *Pending* | *Pending* |

> **Note:** All values will be populated after re-running `model_training.py` with fixed profit column.

---

## 🔍 Known Issues (Resolved)

| Issue | Status | Resolution |
|-------|--------|------------|
| Profit column all zeros | ✅ Resolved | Fixed in `profit_calculator.py`, verified $555M profit |
| `mode_val` reference error | ✅ Resolved | Fixed in `model_training.py` `load_data()` |
| Missing values (NaN) in features | ✅ Resolved | Added `handle_moderate_missing()` to preprocessing |
| Logistic Regression convergence warnings | ⚠️ Acceptable | Fixed with increased `max_iter=2000` and scaling |
| Results summary failing | ⚠️ Pending | Will resolve after re-running training |

---

## 📁 File Structure (Updated)

```
credit-threshold-profit/
├── data/
│   └── processed/
│       ├── cleaned_baseline.parquet (with profit: $555M)
│       ├── cleaned_preprocessed.parquet
│       ├── train_features.parquet
│       ├── test_features.parquet
│       ├── train_engineered_*.parquet (4 plans)
│       └── test_engineered_*.parquet (4 plans)
├── src/
│   ├── baseline_cleaning.py
│   ├── profit_calculator.py
│   ├── data_preprocessing.py
│   ├── data_splitter.py
│   ├── feature_engineering.py
│   └── model_training.py (fixed)
├── reports/
│   ├── data_validation/
│   │   └── validation_report.json
│   └── model_comparison/
│       ├── model_comparison_metrics.csv (pending)
│       ├── profit_curve_*.csv (pending)
│       └── model_comparison.png (pending)
└── TODO.md
```

---

## 🧭 Next Steps (Tomorrow)

1. ✅ **Sleep** (you've earned it!)
2. **Re-run the full pipeline** (after waking up)
3. **Re-run model training** with fixed profit column
4. **Generate final results** and profit curves
5. **Document findings** in README and executive summary
6. **Merge to `main`** and push to GitHub for portfolio

---

## 📌 Current Branch Status

| Branch | Status | Notes |
|--------|--------|-------|
| `feature/complete-model-training` | ✅ Active | Clean, ready for development |
| `initial-build` | ✅ Archived | All work merged to main |
| `main` | ✅ Clean | Ready for final merge |

---

## 🎯 Summary for Tomorrow Morning

| Task | Command |
|------|---------|
| Re-run preprocessing | `python src/data_preprocessing.py` |
| Re-run splitter | `python src/data_splitter.py` |
| Re-run feature engineering | `python src/feature_engineering.py` |
| Re-run model training | `python src/model_training.py` |
| Check profit curves | Look for non-zero values in `reports/model_comparison/` |
| Push results | `git push origin feature/complete-model-training` |
