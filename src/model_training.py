"""
Model Training and Comparison for Credit Threshold Project
Trains Logistic Regression and Random Forest on all 4 feature plans
Compares performance metrics and generates profit curves
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Configurable model trainer for credit threshold project.
    
    Features:
        - Loads all 4 feature engineering plans
        - Trains Logistic Regression and Random Forest on each
        - Performs threshold tuning and profit calculation
        - Compares performance across all configurations
        - Configurable via dictionary
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary with:
                - data_dir: Path to data directory
                - output_dir: Path to output directory
                - feature_plans: List of feature plans to use
                - models: List of models to train ('lr', 'rf')
                - threshold_range: Tuple of (min, max, step)
                - test_size: Fraction of data for testing
                - calibrate: Whether to calibrate probabilities
                - random_state: Random seed for reproducibility
                - n_jobs: Number of parallel jobs
        """
        self.config = config
        self.results = []
        
        # Set default config values
        self.config.setdefault('feature_plans', ['baseline_minimal', 'domain_enhanced', 'interaction_heavy', 'ml_informed'])
        self.config.setdefault('models', ['lr', 'rf'])
        self.config.setdefault('threshold_range', (0.05, 0.95, 0.01))
        self.config.setdefault('calibrate', True)
        self.config.setdefault('random_state', 42)
        self.config.setdefault('n_jobs', -1)
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'reports/model_comparison'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = []
        self.profit_curves = {}
    


    def load_data(self, plan: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load train and test data for a feature plan.
        
        Args:
            plan: Feature plan name
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        data_dir = Path(self.config.get('data_dir', 'data/processed'))
        
        train_path = data_dir / f"train_engineered_{plan}.parquet"
        test_path = data_dir / f"test_engineered_{plan}.parquet"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Engineered data for plan '{plan}' not found")
        
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # --- FIX: Drop all non-numeric columns ---
        string_cols_train = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
        string_cols_test = test_df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        datetime_cols_train = train_df.select_dtypes(include=['datetime64']).columns.tolist()
        datetime_cols_test = test_df.select_dtypes(include=['datetime64']).columns.tolist()
        
        all_non_numeric_train = string_cols_train + datetime_cols_train
        all_non_numeric_test = string_cols_test + datetime_cols_test
        
        if all_non_numeric_train:
            logger.warning(f"Dropping non-numeric columns from train: {all_non_numeric_train}")
            train_df = train_df.drop(columns=all_non_numeric_train)
        
        if all_non_numeric_test:
            logger.warning(f"Dropping non-numeric columns from test: {all_non_numeric_test}")
            test_df = test_df.drop(columns=all_non_numeric_test)
        # --- END FIX ---
        
        # --- FORCE FIX: Impute ALL remaining columns ---
        for col in train_df.columns:
            if train_df[col].isnull().any():
                if train_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    median_val = train_df[col].median()
                    train_df[col] = train_df[col].fillna(median_val)
                else:
                    mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 0
                    train_df[col] = train_df[col].fillna(mode_val)
                logger.debug(f"Imputed {col} with {median_val if 'median' in locals() else mode_val}")
        
        for col in test_df.columns:
            if test_df[col].isnull().any():
                if col in train_df.columns:
                    if train_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        median_val = train_df[col].median()
                        test_df[col] = test_df[col].fillna(median_val)
                    else:
                        mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 0
                        test_df[col] = test_df[col].fillna(mode_val)
                else:
                    if test_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        median_val = test_df[col].median()
                        test_df[col] = test_df[col].fillna(median_val)
                    else:
                        mode_val = test_df[col].mode()[0] if len(test_df[col].mode()) > 0 else 0
                        test_df[col] = test_df[col].fillna(mode_val)
        # --- END FIX ---
        
        # Identify feature columns (exclude target and profit)
        exclude_cols = ['default', 'profit', 'loan_status', 'id']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        X_train = train_df[feature_cols]
        y_train = train_df['default']
        X_test = test_df[feature_cols]
        y_test = test_df['default']
        
        # Store profit values for later use
        self.train_profit = train_df['profit']
        self.test_profit = test_df['profit']
        
        logger.info(f"Loaded {plan}: Train {X_train.shape}, Test {X_test.shape}")
        logger.info(f"Missing values in train: {X_train.isnull().sum().sum()}")
        logger.info(f"Missing values in test: {X_test.isnull().sum().sum()}")
        
        return X_train, X_test, y_train, y_test
    

    
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Train logistic regression model with Platt scaling calibration.
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Logistic Regression...")
        
        # Train base model
        lr = LogisticRegression(
            max_iter=1000,
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        lr.fit(X_train, y_train)
        
        # Calibrate if requested
        if self.config['calibrate']:
            calibrated_lr = CalibratedClassifierCV(
                lr,
                method='sigmoid',
                cv=5,
                n_jobs=self.config['n_jobs']
            )
            calibrated_lr.fit(X_train, y_train)
            model = calibrated_lr
        else:
            model = lr
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        
        return {
            'model': model,
            'y_pred_proba': y_pred_proba,
            'auc': auc,
            'brier': brier,
            'name': 'Logistic Regression'
        }
    
    def train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Train Random Forest with isotonic calibration.
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Random Forest...")
        
        # Train base model
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        rf.fit(X_train, y_train)
        
        # Calibrate if requested
        if self.config['calibrate']:
            calibrated_rf = CalibratedClassifierCV(
                rf,
                method='isotonic',
                cv=5,
                n_jobs=self.config['n_jobs']
            )
            calibrated_rf.fit(X_train, y_train)
            model = calibrated_rf
        else:
            model = rf
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        
        return {
            'model': model,
            'y_pred_proba': y_pred_proba,
            'auc': auc,
            'brier': brier,
            'name': 'Random Forest'
        }
    
    def compute_profit_curve(self, y_pred_proba: np.ndarray, 
                             y_test: pd.Series, profit_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute profit curve for a set of predictions.
        
        Args:
            y_pred_proba: Predicted probabilities
            y_test: True labels
            profit_test: Profit values for test set
            
        Returns:
            Tuple of (thresholds, profits, optimal_threshold)
        """
        min_thresh, max_thresh, step = self.config['threshold_range']
        thresholds = np.arange(min_thresh, max_thresh, step)
        profits = []
        
        for threshold in thresholds:
            # Approve loans with predicted default probability < threshold
            approved_mask = y_pred_proba < threshold
            approved_profits = profit_test[approved_mask]
            
            if len(approved_profits) > 0:
                total_profit = approved_profits.sum()
            else:
                total_profit = 0
            
            profits.append(total_profit)
        
        profits = np.array(profits)
        optimal_idx = np.argmax(profits)
        optimal_threshold = thresholds[optimal_idx]
        optimal_profit = profits[optimal_idx]
        
        return thresholds, profits, optimal_threshold
    
    def train_and_evaluate(self, plan: str, model_type: str) -> Dict:
        """
        Train a model on a feature plan and evaluate.
        
        Args:
            plan: Feature plan name
            model_type: 'lr' or 'rf'
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Training {model_type} on {plan}")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data(plan)
        
        # Train model
        if model_type == 'lr':
            result = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        elif model_type == 'rf':
            result = self.train_random_forest(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compute profit curve
        thresholds, profits, optimal_threshold = self.compute_profit_curve(
            result['y_pred_proba'], y_test, self.test_profit
        )
        
        # Store profit curve
        curve_key = f"{plan}_{model_type}"
        self.profit_curves[curve_key] = {
            'thresholds': thresholds,
            'profits': profits,
            'optimal_threshold': optimal_threshold,
            'optimal_profit': profits[np.argmax(profits)]
        }
        
        # Compile results
        result.update({
            'plan': plan,
            'model_type': model_type,
            'optimal_threshold': optimal_threshold,
            'optimal_profit': profits[np.argmax(profits)],
            'profit_curve': (thresholds, profits)
        })
        
        return result
    
    def run_all(self) -> pd.DataFrame:
        """
        Run training for all feature plans and models.
        
        Returns:
            DataFrame with results
        """
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING AND COMPARISON")
        logger.info("="*60)
        
        results = []
        
        for plan in self.config['feature_plans']:
            for model_type in self.config['models']:
                try:
                    result = self.train_and_evaluate(plan, model_type)
                    results.append(result)
                    
                    logger.info(f"Completed {model_type} on {plan}")
                    logger.info(f"  AUC: {result['auc']:.4f}")
                    logger.info(f"  Brier: {result['brier']:.4f}")
                    logger.info(f"  Optimal threshold: {result['optimal_threshold']:.3f}")
                    logger.info(f"  Optimal profit: ${result['optimal_profit']:,.0f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type} on {plan}: {e}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        # Save results
        self.save_results()
        self.plot_comparison()
        
        logger.info("="*60)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("="*60)
        
        return results_df
    
    def save_results(self) -> None:
        """Save results to files"""
        # Save metrics
        if len(self.results) > 0:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(self.output_dir / 'model_comparison_metrics.csv', index=False)
            
            # Save profit curves
            for key, curve in self.profit_curves.items():
                curve_df = pd.DataFrame({
                    'threshold': curve['thresholds'],
                    'profit': curve['profits']
                })
                curve_df.to_csv(self.output_dir / f'profit_curve_{key}.csv', index=False)
            
            # Save config
            with open(self.output_dir / 'training_config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Results saved to {self.output_dir}")
    
    def plot_comparison(self) -> None:
        """Generate comparison plots"""
        if len(self.results) == 0:
            return
        
        # AUC comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: AUC comparison
        ax = axes[0]
        for model_type in self.config['models']:
            model_results = self.results[self.results['model_type'] == model_type]
            ax.bar(
                model_results['plan'] + f' ({model_type})',
                model_results['auc'],
                alpha=0.7,
                label=model_type
            )
        ax.set_ylabel('AUC')
        ax.set_title('AUC Comparison Across Feature Plans')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Optimal profit comparison
        ax = axes[1]
        for model_type in self.config['models']:
            model_results = self.results[self.results['model_type'] == model_type]
            ax.bar(
                model_results['plan'] + f' ({model_type})',
                model_results['optimal_profit'] / 1e6,
                alpha=0.7,
                label=model_type
            )
        ax.set_ylabel('Optimal Profit ($ Millions)')
        ax.set_title('Optimal Profit Comparison')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Profit curves for best model
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for key, curve in self.profit_curves.items():
            ax.plot(
                curve['thresholds'],
                curve['profits'] / 1e6,
                label=key,
                alpha=0.7
            )
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Profit ($ Millions)')
        ax.set_title('Profit Curves for All Models')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_profit_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plots saved to {self.output_dir}")


def create_training_config(
    data_dir: str = "data/processed",
    output_dir: str = "reports/model_comparison",
    feature_plans: List[str] = None,
    models: List[str] = None,
    threshold_range: Tuple[float, float, float] = (0.05, 0.95, 0.01),
    calibrate: bool = True,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Create configuration dictionary for model training.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        feature_plans: List of feature plans to use
        models: List of models to train ('lr', 'rf')
        threshold_range: Tuple of (min, max, step)
        calibrate: Whether to calibrate probabilities
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        
    Returns:
        Configuration dictionary
    """
    if feature_plans is None:
        feature_plans = ['baseline_minimal', 'domain_enhanced', 'interaction_heavy', 'ml_informed']
    
    if models is None:
        models = ['lr', 'rf']
    
    return {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'feature_plans': feature_plans,
        'models': models,
        'threshold_range': threshold_range,
        'calibrate': calibrate,
        'random_state': random_state,
        'n_jobs': n_jobs
    }


def main():
    """Main execution"""
    # Configure training
    config = create_training_config(
        data_dir="data/processed",
        output_dir="reports/model_comparison",
        feature_plans=['baseline_minimal', 'domain_enhanced', 'interaction_heavy', 'ml_informed'],
        models=['lr', 'rf'],
        threshold_range=(0.05, 0.95, 0.01),
        calibrate=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Run all training
    results_df = trainer.run_all()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    print(results_df[['plan', 'model_type', 'auc', 'brier', 'optimal_threshold', 'optimal_profit']].to_string(index=False))
    print("="*60)
    
    # Save best model
    best_idx = results_df['optimal_profit'].argmax()
    best_model = results_df.iloc[best_idx]
    print(f"\nBest model: {best_model['model_type']} on {best_model['plan']}")
    print(f"Optimal threshold: {best_model['optimal_threshold']:.3f}")
    print(f"Optimal profit: ${best_model['optimal_profit']:,.0f}")


if __name__ == "__main__":
    main()