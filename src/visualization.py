"""
Visualization Module for Credit Threshold Project
Generates additional diagnostic plots for the report
- Calibration curve (reliability diagram)
- Feature importance (LR and RF)
- Confusion matrix at optimal threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
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
MODEL_DIR = Path("D:/credit-threshold-profit/reports/model_comparison")
OUTPUT_DIR = Path("D:/credit-threshold-profit/reports/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best model config (from your results)
BEST_PLAN = "baseline_minimal"
BEST_MODEL_TYPE = "lr"  # logistic regression
BEST_THRESHOLD = 0.620

# ============================================================
# DATA LOADING
# ============================================================

def load_test_data_with_predictions():
    """Load test features, true labels, and predictions for best model"""
    # Load test features
    test_path = DATA_DIR / "test_features.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test features not found: {test_path}")
    
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loaded test features: {len(test_df):,} rows")
    
    # Load predictions for best model
    pred_path = MODEL_DIR / f"predictions_{BEST_PLAN}_{BEST_MODEL_TYPE}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    
    pred_df = pd.read_csv(pred_path)
    logger.info(f"Loaded predictions: {len(pred_df):,} rows")
    
    # Merge on id
    test_df['id'] = test_df['id'].astype(str)
    pred_df['id'] = pred_df['id'].astype(str)
    
    merged = test_df.merge(pred_df[['id', 'default_prob']], on='id', how='inner')
    logger.info(f"Merged: {len(merged):,} rows with predictions")
    
    return merged


# After loading predictions, recompute optimal threshold on test set
def find_test_optimal_threshold(df):
    """Find optimal threshold on test set directly"""
    thresholds = np.arange(0.05, 0.95, 0.01)
    profits = []
    
    for t in thresholds:
        approved = df[df['default_prob'] < t]['profit'].sum()
        profits.append(approved)
    
    profits = np.array(profits)
    optimal_idx = np.argmax(profits)
    optimal_threshold = thresholds[optimal_idx]
    optimal_profit = profits[optimal_idx]
    
    logger.info(f"Test-set optimal threshold: {optimal_threshold:.3f} with profit ${optimal_profit:,.0f}")
    return optimal_threshold, optimal_profit

def load_best_model():
    """Load the best model from disk"""
    model_path = MODEL_DIR / f"model_{BEST_PLAN}_{BEST_MODEL_TYPE}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded model: {model_path.name}")
    return model

def load_feature_names():
    """Get feature names from training data"""
    train_path = DATA_DIR / "train_features.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train features not found: {train_path}")
    
    train_df = pd.read_parquet(train_path)
    # Exclude target and ID columns
    exclude = ['default', 'profit', 'loan_status', 'id']
    feature_cols = [c for c in train_df.columns if c not in exclude]
    logger.info(f"Loaded {len(feature_cols)} feature names")
    return feature_cols

# ============================================================
# VISUALIZATION 1: CALIBRATION CURVE
# ============================================================

def plot_calibration_curve(df):
    """Generate reliability diagram (calibration curve)"""
    logger.info("Generating calibration curve...")
    
    y_true = df['default']
    y_pred = df['default_prob']
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Add histogram of predicted probabilities
    ax2 = ax.twinx()
    ax2.hist(y_pred, bins=10, alpha=0.3, color='blue', label='Prediction Distribution')
    ax2.set_ylabel('Frequency')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curve (Reliability Diagram)\nBest Model: {BEST_PLAN} ({BEST_MODEL_TYPE.upper()})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "calibration_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved calibration curve to {output_path}")
    return output_path

# ============================================================
# VISUALIZATION 2: FEATURE IMPORTANCE (LOGISTIC REGRESSION)
# ============================================================

def plot_feature_importance_lr():
    """Generate feature importance plot for Logistic Regression"""
    logger.info("Generating Logistic Regression feature importance...")
    
    # Load model
    model = load_best_model()
    
    # Get the underlying estimator (CalibratedClassifierCV may wrap the model)
    if hasattr(model, 'estimator'):
        # CalibratedClassifierCV
        lr_model = model.estimator
    else:
        lr_model = model
    
    # Get coefficients
    if not hasattr(lr_model, 'coef_'):
        logger.warning("Model does not have coefficients (not a linear model)")
        return None
    
    coef = lr_model.coef_[0]
    feature_names = load_feature_names()
    
    # Ensure alignment
    if len(coef) != len(feature_names):
        logger.warning(f"Feature mismatch: {len(coef)} coefficients, {len(feature_names)} features")
        return None
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': coef
    })
    
    # Sort by absolute importance
    importance_df['abs_importance'] = importance_df['importance'].abs()
    importance_df = importance_df.sort_values('abs_importance', ascending=False).head(20)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2b6cb0' if x > 0 else '#e53e3e' for x in importance_df['importance']]
    bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coefficient Value')
    ax.set_title(f'Top 20 Feature Importance (Logistic Regression)\n{BEST_PLAN} Plan')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, importance_df['importance']):
        ax.text(val + (0.01 if val > 0 else -0.01), bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "feature_importance_lr.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved LR feature importance to {output_path}")
    return output_path

# ============================================================
# VISUALIZATION 3: FEATURE IMPORTANCE (RANDOM FOREST)
# ============================================================

def plot_feature_importance_rf():
    """Generate feature importance plot for Random Forest (best RF model)"""
    logger.info("Generating Random Forest feature importance...")
    
    # Use the best RF model (baseline_minimal_rf)
    rf_model_path = MODEL_DIR / "model_baseline_minimal_rf.pkl"
    if not rf_model_path.exists():
        logger.warning("Random Forest model not found")
        return None
    
    rf_model = joblib.load(rf_model_path)
    
    # Get feature importances
    if hasattr(rf_model, 'estimator'):
        # CalibratedClassifierCV
        rf_estimator = rf_model.estimator
    else:
        rf_estimator = rf_model
    
    if not hasattr(rf_estimator, 'feature_importances_'):
        logger.warning("Model does not have feature importances")
        return None
    
    importances = rf_estimator.feature_importances_
    feature_names = load_feature_names()
    
    # Ensure alignment
    if len(importances) != len(feature_names):
        logger.warning(f"Feature mismatch: {len(importances)} importances, {len(feature_names)} features")
        return None
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(importance_df['feature'], importance_df['importance'], color='#3182ce')
    
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 20 Feature Importance (Random Forest)\nBaseline Minimal Plan')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, importance_df['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "feature_importance_rf.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved RF feature importance to {output_path}")
    return output_path

# ============================================================
# VISUALIZATION 4: CONFUSION MATRIX AT OPTIMAL THRESHOLD
# ============================================================
def plot_confusion_matrix(df):
    """Generate confusion matrix at optimal threshold (business-focused)"""
    logger.info("Generating confusion matrix...")
    
    y_true = df['default']
    y_pred_proba = df['default_prob']
    
    # Apply optimal threshold
    y_pred = (y_pred_proba >= BEST_THRESHOLD).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Approved (Predicted < 0.620)', 'Declined (Predicted ≥ 0.620)'],
                yticklabels=['Fully Paid (Actual 0)', 'Default (Actual 1)'])
    
    ax.set_xlabel('Predicted Decision')
    ax.set_ylabel('Actual Outcome')
    ax.set_title(f'Business Decisions at Optimal Threshold = {BEST_THRESHOLD:.3f}')
    
    # Get counts
    tn, fp, fn, tp = cm.ravel()
    total_approved = tn + fn
    total_declined = fp + tp
    
    # Add business-focused text box
    metrics_text = f"""
    Total Loans: {tn + fp + fn + tp:,}
    ✅ Approved: {total_approved:,}
    ❌ Declined: {total_declined:,}
    """
    
    ax.text(1.5, 2.5, metrics_text, fontsize=12, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {output_path}")
    logger.info(f"  Approved: {total_approved:,}, Declined: {total_declined:,}")
    
    return output_path

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Generate all visualizations"""
    logger.info("=" * 60)
    logger.info("GENERATING VISUALIZATIONS FOR REPORT")
    logger.info("=" * 60)
    
    # Load data
    df = load_test_data_with_predictions()
    
    # 1. Calibration curve
    plot_calibration_curve(df)
    
    # 2. LR feature importance
    plot_feature_importance_lr()
    
    # 3. RF feature importance
    plot_feature_importance_rf()
    
    # 4. Confusion matrix
    plot_confusion_matrix(df)
    
    logger.info("=" * 60)
    logger.info("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()