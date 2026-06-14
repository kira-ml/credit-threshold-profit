# TODO.md

## Credit Threshold Profit Project

*Last Updated: June 15, 2026, 3:00 AM*

---

## ✅ Completed

| Category | Task | Status |
|----------|------|--------|
| **Data Pipeline** | CSV to Parquet converter | ✅ Done |
| **EDA** | Comprehensive EDA report + 6 visualizations | ✅ Done |
| **Data Cleaning** | Baseline cleaning (1.34M rows, 109 cols) | ✅ Done |
| **Profit Calculation** | Baseline & Advanced profit methods | ✅ Done |
| **Preprocessing** | Sparse columns dropped, categories converted, `grade_num` + `emp_length_num` created | ✅ Done |
| **Data Splitting** | 80/20 train/test split by `issue_d` | ✅ Done |
| **Feature Engineering** | All 4 plans generated (`baseline_minimal`, `domain_enhanced`, `interaction_heavy`, `ml_informed`) | ✅ Done |

---

## 🚧 Current State (as of 3:00 AM)

| Component | Status | Notes |
|-----------|--------|-------|
| **Model Training Script** | ❌ Failing | Imputation bug (`mode_val` reference error) |
| **Logistic Regression** | ❌ Not trained | Fails due to missing values |
| **Random Forest** | ✅ Partially trained | Success on all 4 plans (AUC ~0.999, Brier ~0.002) |
| **Profit Curves** | ✅ Generated for RF | Optimal threshold consistently 0.94 |
| **Results DataFrame** | ❌ Empty | No results from LR, so summary fails |

---

## 🔴 Immediate Fixes (Next Session)

| # | Task | Priority | Location | Est. Time |
|---|------|----------|----------|-----------|
| 1 | Fix `mode_val` reference error in imputation | 🔴 Critical | `model_training.py` (load_data) | 5 min |
| 2 | Re-run model training with fix | 🔴 Critical | CLI | 20-30 min |
| 3 | Validate Logistic Regression training | 🔴 Critical | `model_training.py` | 10 min |
| 4 | Generate final results summary | 🔴 Critical | `model_training.py` | 5 min |

---

## 🟡 Pending Tasks (After Training Works)

| # | Task | Priority | Location | Est. Time |
|---|------|----------|----------|-----------|
| 1 | Compare AUC, Brier, optimal threshold across all 8 model/plan combos | 🟠 Medium | `model_training.py` | 10 min |
| 2 | Identify best model + feature plan | 🟠 Medium | Analysis | 10 min |
| 3 | Bootstrap confidence intervals for optimal profit | 🟠 Medium | New script | 30 min |
| 4 | Stress test: shift default probability by ±20% | 🟠 Medium | New script | 30 min |

---

## 🔵 Final Deliverables (After Training Complete)

| # | Task | Priority | Est. Time |
|---|------|----------|-----------|
| 1 | Create final profit curve visualization (publication quality) | 🔴 High | 20 min |
| 2 | Write executive summary (1 paragraph, business impact in $) | 🔴 High | 20 min |
| 3 | Update README.md with results and profit curve | 🟠 Medium | 20 min |
| 4 | Push to GitHub | 🟠 Medium | 10 min |

---

## 📊 Current Metrics (Partial - RF Only)

| Feature Plan | AUC | Brier | Optimal Threshold | Optimal Profit |
|--------------|-----|-------|------------------|----------------|
| baseline_minimal | 0.9998 | 0.0023 | 0.940 | $97,112,912 |
| domain_enhanced | 0.9999 | 0.0018 | 0.940 | $96,986,059 |
| interaction_heavy | 0.9999 | 0.0020 | 0.940 | $97,156,086 |
| ml_informed | *Pending* | *Pending* | *Pending* | *Pending* |

> **Note:** Logistic Regression results are missing due to imputation bug.

---

## 🔍 Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| `mode_val` referenced before assignment | ❌ Unresolved | Fix in `load_data()` |
| Missing values (NaN) in features | ❌ Unresolved | Imputation fix should resolve |
| Logistic Regression failing | ❌ Unresolved | Requires fixed imputation |
| Results summary failing | ❌ Unresolved | Requires at least one successful training |

---

## 📁 File Structure

```
credit-threshold-profit/
├── data/
│   └── processed/
│       ├── cleaned_baseline.parquet
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
│   └── model_training.py
├── reports/
│   └── model_comparison/
│       ├── model_comparison_metrics.csv (partial)
│       ├── profit_curve_*.csv (RF only)
│       └── model_comparison.png (partial)
└── TODO.md
```

---

## 🧭 Next Steps (High Level)

1. **Fix imputation bug** in `model_training.py`
2. **Re-run model training** to train all 8 combinations
3. **Generate final results** and profit curves
4. **Document findings** in README and executive summary
5. **Push to GitHub** for portfolio
