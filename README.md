# Loan Profit Prediction

**Predicting expected loan profitability to inform approval decisions that maximize portfolio returns.**

---

## 1. Problem Framing

### What This Project Addresses

Consumer lenders typically build models to predict whether a borrower will default. Default probability alone does not capture the full economic picture: a high-interest loan to a medium-risk borrower may generate more profit than a low-interest loan to a very safe borrower.

This project reframes the problem as a regression task: predict the **expected net present value (NPV)** of a loan's cash flows using only information available at application. The resulting model informs which loans to approve to maximize portfolio profitability.

### Business Context

A medium-sized consumer lender issues hundreds of thousands of unsecured personal loans annually. Credit risk teams, the CFO, and product managers jointly set approval policies. Current decisions often rely on a default probability score, which overlooks the fact that some higher-risk loans are still profitable when priced appropriately, while some ultra-safe loans return little interest relative to operating costs.

### Why It Matters

Profitability-driven underwriting directly affects net interest income, portfolio return on assets, and capital allocation. A 1–2% improvement in profit per loan can translate into tens of millions of dollars for a lender of scale. Shifting the objective from minimizing defaults to maximizing profit aligns modeling work with the income statement.

### Expected Value

A model that ranks loans by expected profit supports a profit-threshold approval rule. In a hold-out simulation, such a rule is expected to increase aggregate portfolio profit by a measurable percentage compared to a default-rate-based policy.

---

## 2. Business Objective

**Primary goal:** Maximize the total net present value of the loan portfolio by accepting only applications whose predicted profit exceeds a defined threshold.

**Key stakeholders:**
- Credit Risk Team (sets risk appetite and validation standards)
- Chief Financial Officer (accountable for net interest income and return on assets)
- Product Managers (design loan pricing and approval rules)

**Decisions supported:**
- Which loan applications to approve (accept/reject) based on a profit cut-off
- Pricing adjustments for borderline applicants (enabled but beyond current scope)

**Expected business impact:** An estimated uplift in portfolio-level NPV of 3–8% on out-of-time data, relative to an approve-all policy or a default-probability cut-off, while keeping the total number of loans comparable.

---

## 3. Machine Learning Objective

**Prediction target:** Ex-post loan-level NPV of cash flows, measured in USD. Aggregates discounted principal, interest payments, recoveries, and charge-offs over the loan's life.

**Input features:** All information available at origination — loan amount, interest rate, term, purpose, debt-to-income ratio, employment length, annual income, FICO score (or grade), number of open credit lines, revolving utilization, home ownership, and related attributes.

**Task type:** Supervised regression.

**Model output:** For each loan application, a scalar predicted NPV. Predictions are used to rank applications and apply an approval rule: accept if predicted profit exceeds a threshold θ.

---

## 4. Assumptions

**Data assumptions:**
- The Lending Club dataset is representative of a typical unsecured consumer loan portfolio
- Loan performance after maturity is fully observed (all loans are paid off, charged off, or reached full term)
- A risk-free discount rate (e.g., 2%) approximates the lender's cost of funds
- Macroeconomic conditions during the test period are similar enough to the training period that relationships remain stable

**Domain assumptions:**
- Loan profitability is adequately captured by the NPV of contractual and realized cash flows, ignoring fixed operational costs per application
- The lender's approval decision is binary and does not alter loan terms; pricing is pre-set
- A declined application yields exactly zero profit

**Modeling assumptions:**
- The relationship between application features and realized profit can be learned from historical data
- The profit target, though computed ex post, is a valid proxy for economic value at origination
- Models are trained only on funded loans; selection bias from past approvals is acknowledged but accepted for this scope

---

## 5. Constraints

- **Hardware:** Must run on a standard Intel Core i5 laptop with 16 GB RAM; training time should be measured in minutes, not hours
- **Data:** Only publicly available Lending Club data (pre-COVID snapshot) is used; no external credit bureau or macroeconomic features
- **Time:** The full project should be completable within 2–3 weeks of part-time effort
- **Scope:** No deployment, API, or MLOps work; no causal inference or pricing optimization; no deep learning or GPU-dependent models

---

## 6. Success Metrics

### Business Metrics

| Metric | Description |
|--------|-------------|
| Aggregate portfolio profit | Sum of realized NPV for all loans accepted under a policy, computed on the held-out test period |
| Profit uplift vs. baselines | Percentage increase over (a) approve-all and (b) a default-probability cut-off with similar acceptance rate |
| Profit per accepted loan | Average realized NPV of accepted applications |
| Acceptance rate | Fraction of applications approved; monitored to ensure uplift isn't achieved by simply taking far fewer loans |

### Model Metrics

| Metric | Role |
|--------|------|
| RMSE (primary) | Penalizes large errors that can distort approval decisions; measured in dollars |
| MAE (secondary) | Directly interpretable dollar error per loan |
| R² | Proportion of variance in realized profit explained; compares against a constant mean predictor |
| Spearman rank correlation | Evaluates how well the model orders loans by profitability, crucial for threshold-based decisions |

RMSE is used because the cost of mis-predicting profit is roughly quadratic. Spearman captures ranking quality needed for the final decision rule.

---

## 7. Evaluation Strategy

**Validation methodology:** Temporal (out-of-time) split. Data are ordered by loan issue date. The earliest 60% form the training set, the next 20% the validation set, and the most recent 20% the test set.

**Train/validation/test roles:**
- Training: feature selection, coefficient estimation, hyperparameter tuning
- Validation: model complexity selection and profit-threshold (θ) optimization
- Test: held out until final evaluation; used to report all metrics and simulate the profit-based approval policy

**Cross-validation:** Expanding-window time-series cross-validation on the training set (e.g., 5 folds, each adding one year of data).

**Leakage prevention:**
- No feature uses information unavailable at origination
- Target NPV is computed from actual outcomes but never seen during prediction
- All transformations are fit on training data only, then applied to validation and test

**Baseline comparison:** All baselines are evaluated on the same test period using identical profit simulation. The profit threshold for each model is optimized on validation data.

---

## 8. Baseline Solutions (Implemented First)

### Baseline 1: Historical Mean Predictor

Always predict the average realized profit of all loans in the training set.

**Why start here:** Sets a lower bound any useful model must beat. Validates the data pipeline and profit simulation.

**Advantages:** No training cost, interpretable, reality check on predictability.

**Limitations:** No discrimination between loans; fails to improve portfolio selection.

---

### Baseline 2: Linear Regression (Ridge-regularized)

**Why start here:** Standard regression baseline. Reveals linear relationships between features and profit. Trains quickly. If it performs comparably to more complex models, it is the preferred solution.

**Advantages:** Fast, interpretable coefficients, straightforward residual diagnostics, low overfitting risk with regularization.

**Limitations:** Cannot capture non-linear effects or interactions without manual feature engineering. Sensitive to outliers.

---

### Baseline 3: Shallow Decision Tree Regressor (max_depth=5)

**Why start here:** Captures simple non-linearities and interactions that linear regression might miss. Remains interpretable. Provides a benchmark for tree-based models.

**Advantages:** Handles non-linear relationships automatically, produces intuitive rules, no feature scaling required.

**Limitations:** High variance, prone to overfitting if depth is not constrained, may underfit the continuous profit surface if kept too shallow.

---

## 9. Advanced Approaches (Only If Needed)

### Advanced Method 1: Random Forest Regressor

**When to consider:** The decision tree baseline shows non-linear patterns but suffers from high variance, and validation RMSE improves meaningfully over linear regression while remaining stable.

**Benefits:** Reduces variance through bagging, captures feature interactions automatically, provides feature importance estimates.

**Risks:** Slower training, less interpretable, requires hyperparameter tuning, can overfit with noisy targets.

---

### Advanced Method 2: Gradient Boosted Trees (XGBoost/LightGBM)

**When to consider:** Further improvement in RMSE and Spearman rank correlation on the validation set, and the business simulation shows meaningful profit uplift from improved rankings.

**Benefits:** Often strong predictive accuracy, built-in regularization, handles missing values natively, memory-efficient.

**Risks:** Many hyperparameters, tuning is time-consuming, black-box nature requires extra explanation effort, marginal gains may not justify added complexity.

---

### Advanced Method 3: Feature-Engineered Linear Model with Interaction Terms

**When to consider:** The linear regression baseline is already competitive and domain knowledge suggests specific interactions (e.g., grade × interest rate, loan amount × DTI).

**Benefits:** Maintains full transparency, can capture non-linear effects, low computational cost.

**Risks:** Overfitting risk if too many interactions are added, labor-intensive feature construction, limited gains if the true relationship is highly non-smooth.

---

## 10. Escalation Logic

The progression from mean predictor → linear regression → decision tree → ensemble methods follows an evidence-based path:

- **Assumption testing:** The mean predictor tests basic predictability. Linear regression tests linearity. Residual plots reveal non-linear patterns that motivate tree-based models.
- **Bias-variance trade-off:** Linear regression has low variance but potentially high bias. Trees reduce bias but increase variance. Random Forest reduces variance. Boosting further reduces bias. The progression balances this trade-off step by step.
- **Interpretability:** Simple, interpretable models come first. Complexity is added when evidence shows it improves profit-based decisions, and interpretability tools (feature importance, SHAP) are applied.
- **Computational efficiency:** Each step increases computation modestly. Tuning a booster is only undertaken if simpler models leave substantial profit on the table.
- **Overfitting risk:** Temporal cross-validation is used at each stage. If a more complex model improves validation profit but degrades on the test set, the previous model is retained.

---

## 11. Common Mistakes to Avoid

- Predicting default probability instead of profit — reverts to the conventional problem
- Using random train/test splits instead of temporal splits — leaks future information
- Over-optimizing R² without checking whether improved rankings translate to higher portfolio profit
- Adding complex models without first establishing that simpler baselines are inadequate
- Forgetting to discount cash flows — distorts the profit target

---

## 12. Recommended Scope and Progression

**Scope:** Build an expected-loan-profit model using Lending Club data with a computed NPV target. Evaluate through a profit-threshold simulation on a temporal hold-out set. Keep the modeling progression need-driven.

**Progression:**
1. Historical mean predictor (sanity check)
2. Linear regression with Ridge regularization, using 15–20 clean origination features
3. Shallow decision tree (max_depth ≤ 5) as a non-linear benchmark
4. If profit uplift is marginal, iterate with feature engineering on the linear model
5. Only if RMSE and ranking metrics improve notably and the profit simulation shows meaningful dollar uplift, experiment with Random Forest and LightGBM

---

## Repository Structure

```
loan-profit-prediction/
├── data/
│   ├── raw/              # Original Lending Club CSV files
│   └── processed/        # Cleaned data, engineered features
├── notebooks/
│   └── 01_initial_eda.ipynb
├── src/
│   ├── data_prep.py      # Target NPV calculation, cleaning
│   ├── features.py       # Feature engineering
│   └── evaluation.py     # Profit simulation, metrics
├── models/               # Saved model files
├── outputs/              # Figures, tables, result summaries
├── README.md
└── requirements.txt
```

## Data

Public Lending Club loan data (available on Kaggle), containing application features, loan terms, interest rates, and monthly payment outcomes including recoveries and charge-offs.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
```