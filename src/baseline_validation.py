"""
Baseline Validation for Credit Threshold Project
Compares model performance against simple baselines and domain heuristics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("D:/credit-threshold-profit/data/processed")
REPORTS_DIR = Path("D:/credit-threshold-profit/reports/model_comparison")

# ============================================================
# DATA LOADING
# ============================================================

def load_test_data():
    """Load test data with profit and features"""
    # Try to load test_features.parquet first
    path = DATA_DIR / "test_features.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded test features: {len(df):,} rows")
    else:
        # Fallback to cleaned_preprocessed
        path = DATA_DIR / "cleaned_preprocessed.parquet"
        df = pd.read_parquet(path)
        logger.info(f"Loaded preprocessed data: {len(df):,} rows")
    
    # Ensure id column is string type for merging
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
    
    return df

def load_model_predictions():
    """Load model predictions from best model"""
    # Best model: Logistic Regression on baseline_minimal
    pred_path = REPORTS_DIR / "predictions_baseline_minimal_lr.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        logger.info(f"Loaded predictions: {len(df):,} rows")
        
        # Ensure id column is string type for merging
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
        
        return df
    else:
        logger.warning("No prediction files found. Run model_training.py with save_predictions=True")
        return None

# ============================================================
# BASELINE STRATEGIES
# ============================================================

def baseline_approve_all(df):
    """Approve all loans - upper bound"""
    return df['profit'].sum()

def baseline_reject_all(df):
    """Reject all loans - lower bound"""
    return 0

def baseline_random(df, p=0.5, n_runs=10):
    """Random approval with probability p"""
    profits = []
    for seed in range(n_runs):
        np.random.seed(seed)
        mask = np.random.random(len(df)) < p
        profit = df[mask]['profit'].sum()
        profits.append(profit)
    return np.mean(profits), np.std(profits)

def baseline_fixed_threshold(df, threshold, prob_col='default_prob'):
    """Fixed probability threshold"""
    if prob_col not in df.columns:
        logger.warning(f"Column '{prob_col}' not found. Cannot compute threshold baseline.")
        return None
    approved = df[df[prob_col] <= threshold]
    return approved['profit'].sum()

def baseline_grade(df):
    """Reject loans with grade D, E, F, G"""
    if 'grade' not in df.columns:
        logger.warning("'grade' column not found. Cannot compute grade baseline.")
        return None
    approved = df[~df['grade'].isin(['D', 'E', 'F', 'G'])]
    return approved['profit'].sum()

def baseline_dti(df, max_dti=30):
    """Reject loans with DTI > max_dti"""
    if 'dti' not in df.columns:
        logger.warning("'dti' column not found. Cannot compute DTI baseline.")
        return None
    approved = df[df['dti'] <= max_dti]
    return approved['profit'].sum()

def baseline_fico(df, min_fico=660):
    """Reject loans with FICO < min_fico"""
    if 'fico_range_low' not in df.columns:
        logger.warning("'fico_range_low' column not found. Cannot compute FICO baseline.")
        return None
    approved = df[df['fico_range_low'] >= min_fico]
    return approved['profit'].sum()

# ============================================================
# MAIN VALIDATION
# ============================================================

def run_baseline_validation():
    """Run all baselines and return results"""
    results = {}
    
    # Load data
    test_df = load_test_data()
    pred_df = load_model_predictions()
    
    # If predictions exist, merge them into test data
    if pred_df is not None:
        # Ensure both dataframes have id as string
        test_df['id'] = test_df['id'].astype(str)
        pred_df['id'] = pred_df['id'].astype(str)
        
        test_df = test_df.merge(pred_df[['id', 'default_prob']], on='id', how='left')
        logger.info(f"Merged predictions: {test_df['default_prob'].notna().sum():,} rows have predictions")
    
    # 1. Approve-All
    results['approve_all'] = baseline_approve_all(test_df)
    logger.info(f"Approve-All: ${results['approve_all']:,.0f}")
    
    # 2. Reject-All
    results['reject_all'] = baseline_reject_all(test_df)
    logger.info(f"Reject-All: ${results['reject_all']:,.0f}")
    
    # 3. Random Approval
    random_mean, random_std = baseline_random(test_df, p=0.5, n_runs=10)
    results['random_mean'] = random_mean
    results['random_std'] = random_std
    logger.info(f"Random (50%): ${random_mean:,.0f} ±${random_std:,.0f}")
    
    # 4. Fixed thresholds (if predictions available)
    if 'default_prob' in test_df.columns:
        for t in [0.30, 0.50, 0.70]:
            profit = baseline_fixed_threshold(test_df, t)
            results[f'threshold_{t:.2f}'] = profit
            logger.info(f"Fixed Threshold {t:.2f}: ${profit:,.0f}")
    else:
        logger.warning("No default_prob column. Skipping threshold baselines.")
    
    # 5. Domain heuristics
    grade_profit = baseline_grade(test_df)
    if grade_profit is not None:
        results['grade_based'] = grade_profit
        logger.info(f"Grade-Based (D-F): ${grade_profit:,.0f}")
    
    dti_profit = baseline_dti(test_df)
    if dti_profit is not None:
        results['dti_based'] = dti_profit
        logger.info(f"DTI-Based (≤30%): ${dti_profit:,.0f}")
    
    fico_profit = baseline_fico(test_df)
    if fico_profit is not None:
        results['fico_based'] = fico_profit
        logger.info(f"FICO-Based (≥660): ${fico_profit:,.0f}")
    
    # 6. Your model's performance (for comparison)
    if 'default_prob' in test_df.columns:
        model_profit = baseline_fixed_threshold(test_df, 0.620)  # Optimal threshold
        results['model_best'] = model_profit
        logger.info(f"Your Model (threshold 0.620): ${model_profit:,.0f}")
    
    return results

def format_results(results):
    """Format results for display"""
    print("\n" + "="*80)
    print("BASELINE VALIDATION RESULTS")
    print("="*80)
    print(f"{'Baseline':<30} {'Profit':>20} {'Lift vs Approve-All':>20}")
    print("-"*80)
    
    approve_all = results.get('approve_all', 0)
    
    for k, v in results.items():
        if v is None:
            continue
        if k == 'random_std':
            continue
        if k == 'random_mean':
            lift = ((v - approve_all) / approve_all * 100) if approve_all != 0 else 0
            print(f"{'Random (50%)':<30} ${v:>19,.0f} ±${results['random_std']:,.0f} {lift:>20.1f}%")
        elif k == 'approve_all':
            print(f"{'Approve-All':<30} ${v:>19,.0f} {'—':>20}")
        elif k == 'reject_all':
            lift = ((v - approve_all) / approve_all * 100) if approve_all != 0 else 0
            print(f"{'Reject-All':<30} ${v:>19,.0f} {lift:>20.1f}%")
        elif k.startswith('threshold_'):
            t = k.split('_')[1]
            lift = ((v - approve_all) / approve_all * 100) if approve_all != 0 else 0
            print(f"{f'Fixed Threshold {t}':<30} ${v:>19,.0f} {lift:>20.1f}%")
        elif k == 'grade_based':
            lift = ((v - approve_all) / approve_all * 100) if approve_all != 0 else 0
            print(f"{'Grade-Based (D-F)':<30} ${v:>19,.0f} {lift:>20.1f}%")
        elif k == 'dti_based':
            lift = ((v - approve_all) / approve_all * 100) if approve_all != 0 else 0
            print(f"{'DTI-Based (≤30%)':<30} ${v:>19,.0f} {lift:>20.1f}%")
        elif k == 'fico_based':
            lift = ((v - approve_all) / approve_all * 100) if approve_all != 0 else 0
            print(f"{'FICO-Based (≥660)':<30} ${v:>19,.0f} {lift:>20.1f}%")
        elif k == 'model_best':
            lift = ((v - approve_all) / approve_all * 100) if approve_all != 0 else 0
            print(f"{'Your Model (optimal)':<30} ${v:>19,.0f} {lift:>20.1f}%")
        else:
            lift = ((v - approve_all) / approve_all * 100) if approve_all != 0 else 0
            print(f"{k:<30} ${v:>19,.0f} {lift:>20.1f}%")
    print("="*80)

def main():
    """Run baseline validation and save results"""
    results = run_baseline_validation()
    
    # Display results
    format_results(results)
    
    # Save results
    output_path = Path("reports/baseline_validation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(results).to_csv(output_path)
    logger.info(f"Results saved to {output_path}")
    
    # Save comparison plot
    save_comparison_plot(results)

def save_comparison_plot(results):
    """Save a bar chart comparing baselines"""
    import matplotlib.pyplot as plt
    
    # Prepare data
    labels = []
    values = []
    colors = []
    
    for k, v in results.items():
        if v is None or k == 'random_std':
            continue
        labels.append(k.replace('_', ' ').title())
        values.append(v / 1e6)  # Convert to millions
        if k == 'model_best':
            colors.append('green')
        elif k == 'approve_all':
            colors.append('blue')
        elif k == 'reject_all':
            colors.append('red')
        else:
            colors.append('gray')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.axhline(results['model_best'] / 1e6, color='green', linestyle='--', label='Your Model')
    plt.xlabel('Baseline Strategy')
    plt.ylabel('Profit ($ Millions)')
    plt.title('Baseline Comparison: Your Model vs. Simple Strategies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    output_path = Path("reports/baseline_comparison.png")
    plt.savefig(output_path, dpi=150)
    logger.info(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    main()