# credit-threshold-profit

## Profit-Driven Credit Approval Thresholding: From Predictions to Portfolio Outcomes

A data science project that asks a simple question: *which loan applicants should a lender approve to maximize net profit?* The answer isn't found in a better model — it's found in a better threshold.

---

## 📊 Key Results (Test Set: 2016–2018)

| Metric | Value |
| :--- | :--- |
| **Best Model** | Logistic Regression (AUC: 0.6671) |
| **Optimal Approval Threshold** | 0.620 |
| **Approve-All Strategy (Baseline)** | -$331.9M (Loss) |
| **Optimal Threshold Strategy** | -$331.6M (Loss) |
| **Loss Reduction vs. Approve-All** | **+$280k** |

> **Note:** The 2016–2018 period had negative returns overall. The optimal threshold reduced portfolio losses by $280k compared to an approve-all strategy, validating the framework even in adverse credit cycles.

**[View the Full Academic Report](credit_threshold_profit_report.pdf)** | **[View the LinkedIn Executive Brief](credit_threshold_linkedin_brief.pdf)**

---

## The Problem

Credit models typically predict the probability that a borrower will default. But a prediction alone doesn't make a decision. Somewhere, someone picks a cutoff: above this probability, reject. Below it, approve.

Most default models are evaluated by how well they separate good loans from bad ones (AUC, accuracy). That's a statistical question. The business question is different: **which threshold makes the most money?**

A marginal loan at 22% default probability might generate substantial interest income that outweighs the expected loss. A conservative cutoff at 15% might reject profitable borrowers. An aggressive cutoff at 35% might approve too many loans that default. The right answer depends on the actual dollars involved — not just the probabilities.

This project explores that tradeoff using real consumer loan data.

---

## What This Project Does

Using publicly available LendingClub loan data, this project:

1. Trains a probabilistic classifier to estimate default likelihood for each loan
2. Simulates what would have happened under different approval thresholds
3. Calculates the total net cash flow (`total_received - funded_amount`) at each threshold
4. Identifies the threshold that maximizes portfolio value

The core output is a **profit curve** — a single visualization showing how portfolio-level net return changes as the approval cutoff moves from conservative to aggressive, with the optimal point clearly marked.

---

## Why This Framing Matters

| Typical Student Project | This Project |
|-------------------------|--------------|
| "Model achieved 0.82 AUC" | "At this threshold, loss reduced by $280k" |
| Optimizes for accuracy | Optimizes for a business outcome |
| Stops at probability predictions | Asks what happens when predictions become decisions |
| ROC curve is the final deliverable | Profit curve is the final deliverable |

The shift is from asking "how good is the model?" to asking "what value does the model create?"

---

## Project Scope

This project is a **retrospective simulation** using historical data where loan outcomes are already known. It demonstrates a framework for threshold optimization — not a live decision system.

**In scope:**
- Probability of default modeling with calibrated estimates
- Threshold sweeping and portfolio-level profit calculation
- Comparison against 8 baseline strategies (approve-all, fixed-threshold, domain heuristics)
- Target leakage detection and removal

**Out of scope:**
- Predicting loss given default or exposure at default separately (realized cash flows are used)
- Macroeconomic forecasting or multi-period portfolio dynamics
- Interest rate optimization or pricing models
- Real-time deployment, APIs, or dashboards
- Regulatory capital calculations

---

## 🧠 Hard Lessons Learned (Data Integrity)

**Target Leakage Detection:**
Initial models achieved an **AUC of 0.9999**—a clear sign of target leakage. I identified 25 payment/collection columns (e.g., `total_pymnt`, `recoveries`, `last_pymnt_d`) that contained post-origination information.

**Resolution:**
I implemented a pattern-based removal function to systematically exclude these features. After removal, AUC dropped from **0.9999 to a realistic 0.6671**, confirming the leakage was effectively fixed.

This experience reinforced that data integrity is just as important as model selection.

---

## Approach

### Baseline Solutions (implemented first)

**1. Approve-Everything Strategy**
Approve all loans, compute total net profit. Establishes a lower bound and quantifies what happens with no selection at all.

**2. Logistic Regression with Threshold Tuning**
A transparent linear model on a small set of interpretable features. Threshold is chosen to maximize profit on validation data. Simple, fast, and explainable.

**3. Random Forest with Isotonic Calibration**
Captures non-linear relationships. Isotonic calibration ensures probabilities are reliable before threshold optimization. Represents a solid, practical baseline.

### Advanced Extensions (built only after baselines are solid)

**1. LightGBM with Nested Cross-Validation**
Nested CV provides an unbiased estimate of out-of-sample profit performance, with bootstrap confidence intervals around the optimal threshold.

**2. Profit-Weighted Training**
Samples are weighted by their realized dollar outcome during training, aligning the model's objective more directly with the business goal before threshold tuning.

**3. Stress Test: Hypothetical Default Rate Shift**
Simulates a recession-like environment by shifting default log-odds upward, then recomputes the profit curve to see how fragile (or robust) the optimal threshold is.

---

## Evaluation

The primary metric is **net profit lift** over baseline strategies, measured on a held-out test set. Confidence intervals come from bootstrapping.

Calibration quality is checked using:
- Brier score
- Reliability diagrams

A good model with poorly calibrated probabilities can produce a misleading profit curve. Calibration is assessed before any threshold decisions are made.

---

## Limitations

- This is a **retrospective analysis**. Realized outcomes are used to calculate profit; in practice, future outcomes are uncertain.
- The optimal threshold is conditional on the historical data distribution. If the applicant pool changes, the threshold may need updating.
- Operational costs (underwriting, collections) and capital constraints are not modeled.
- LendingClub data reflects a specific period and borrower population; results don't generalize to all lending contexts without careful consideration.

Acknowledging these constraints is important. The value here is in the framework and the thinking, not in claiming to have found a universally optimal cutoff.

---

## Why This Project Exists

The goal is to demonstrate how predictive models translate into tangible financial value—proving that a well-calibrated linear model can outperform a complex black-box when the objective is profit, not accuracy.

---

## Setup

```bash
git clone https://github.com/kira-ml/credit-threshold-profit.git
cd credit-threshold-profit
pip install -r requirements.txt
```

Run the main analysis:
```bash
python src/model_training.py
python src/baseline_validation.py
python src/generate_report.py
python src/generate_summary.py
```

---

## 📂 File Structure

```
credit-threshold-profit/
├── credit_threshold_profit_report.pdf      # 15-page academic report
├── credit_threshold_linkedin_brief.pdf     # 4-page LinkedIn executive brief
├── README.md
├── TODO.md
├── data/
│   └── processed/                          # Engineered Parquet files
├── src/
│   ├── baseline_cleaning.py
│   ├── profit_calculator.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── baseline_validation.py
│   ├── generate_report.py
│   ├── generate_summary.py
│   └── visualization.py
└── reports/
    ├── model_comparison/                   # Metrics, predictions, models
    └── visualizations/                     # Calibration curve, confusion matrix
```

---

## License

MIT
