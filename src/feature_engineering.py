"""
Safe Feature Engineering for Credit Threshold Project
Prevents data leakage by calculating statistics on training data only
and applying them to test data without re-computation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SafeFeatureEngineer:
    """
    Feature engineering with strict train/test separation.
    
    All aggregations (means, default rates, etc.) are calculated on
    training data and applied to test data to prevent leakage.
    """
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Initialize with train and test DataFrames.
        
        Args:
            train_df: Training data (pre-split)
            test_df: Test data (pre-split)
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.train_features = []
        self.test_features = []
        
        # Store statistics learned from training data
        self.statistics = {}
    
    def _validate_columns(self, df: pd.DataFrame, required: List[str]) -> bool:
        """Check if required columns exist"""
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True
    
    def _apply_mapping(self, df: pd.DataFrame, col: str, mapping: Dict) -> pd.DataFrame:
        """Apply a mapping to a column safely"""
        df = df.copy()
        df[col] = df[col].map(mapping).fillna(0)
        return df
    
    def _add_feature(self, df: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
        """Add a feature to a DataFrame"""
        df = df.copy()
        df[name] = values
        return df
    
    def apply_base_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply base features that don't require statistics.
        These can be applied directly to both train and test.
        """
        logger.info("Applying base features...")
        
        # --- FIX: Convert term to numeric ---
        term_map = {
            ' 36 months': 36, '36 months': 36,
            ' 60 months': 60, '60 months': 60
        }
        self.train_df['term_numeric'] = self.train_df['term'].map(term_map).fillna(0)
        self.test_df['term_numeric'] = self.test_df['term'].map(term_map).fillna(0)
        
        # Drop original term column
        self.train_df = self.train_df.drop(columns=['term'], errors='ignore')
        self.test_df = self.test_df.drop(columns=['term'], errors='ignore')
        
        # Rename term_numeric to term
        self.train_df = self.train_df.rename(columns={'term_numeric': 'term'})
        self.test_df = self.test_df.rename(columns={'term_numeric': 'term'})
        
        # --- Continue with existing features ---
        # Loan-to-income ratio
        self.train_df = self._add_feature(
            self.train_df, 'loan_to_income',
            self.train_df['funded_amnt'] / (self.train_df['annual_inc'] + 1)
        )
        self.test_df = self._add_feature(
            self.test_df, 'loan_to_income',
            self.test_df['funded_amnt'] / (self.test_df['annual_inc'] + 1)
        )
        
        # Payment-to-income ratio
        self.train_df = self._add_feature(
            self.train_df, 'payment_to_income',
            self.train_df['installment'] / (self.train_df['annual_inc'] / 12 + 1)
        )
        self.test_df = self._add_feature(
            self.test_df, 'payment_to_income',
            self.test_df['installment'] / (self.test_df['annual_inc'] / 12 + 1)
        )
        
        # Term flag (60 months) - already numeric from the conversion above
        self.train_df = self._add_feature(
            self.train_df, 'term_60',
            (self.train_df['term'] == 60).astype(int)
        )
        self.test_df = self._add_feature(
            self.test_df, 'term_60',
            (self.test_df['term'] == 60).astype(int)
        )
        
        # Home ownership flags
        self.train_df = self._add_feature(
            self.train_df, 'home_ownership_rent',
            (self.train_df['home_ownership'] == 'RENT').astype(int)
        )
        self.test_df = self._add_feature(
            self.test_df, 'home_ownership_rent',
            (self.test_df['home_ownership'] == 'RENT').astype(int)
        )
        
        self.train_df = self._add_feature(
            self.train_df, 'home_ownership_mortgage',
            (self.train_df['home_ownership'] == 'MORTGAGE').astype(int)
        )
        self.test_df = self._add_feature(
            self.test_df, 'home_ownership_mortgage',
            (self.test_df['home_ownership'] == 'MORTGAGE').astype(int)
        )
        
        # Employment length numeric
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years': 8, '9 years': 9, '10+ years': 10
        }
        self.train_df = self._apply_mapping(self.train_df, 'emp_length_num', emp_map)
        self.test_df = self._apply_mapping(self.test_df, 'emp_length_num', emp_map)
        
        # Grade numeric
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        self.train_df = self._apply_mapping(self.train_df, 'grade_num', grade_map)
        self.test_df = self._apply_mapping(self.test_df, 'grade_num', grade_map)
        
        # Credit utilization (0-1 scale)
        self.train_df = self._add_feature(
            self.train_df, 'utilization',
            self.train_df['revol_util'].fillna(0) / 100
        )
        self.test_df = self._add_feature(
            self.test_df, 'utilization',
            self.test_df['revol_util'].fillna(0) / 100
        )
        
        # FICO normalized
        self.train_df = self._add_feature(
            self.train_df, 'fico_norm',
            (self.train_df['fico_range_low'].fillna(600) - 600) / 250
        )
        self.test_df = self._add_feature(
            self.test_df, 'fico_norm',
            (self.test_df['fico_range_low'].fillna(600) - 600) / 250
        )
        
        logger.info("Base features applied to train and test")
        return self.train_df, self.test_df
    
    def apply_domain_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply domain-enhanced features.
        Statistics are calculated on train only, then applied to test.
        """
        logger.info("Applying domain-enhanced features...")
        
        # Rate spread: difference from grade average (calculated on train)
        grade_avg_int = self.train_df.groupby('grade')['int_rate'].mean().to_dict()
        self.statistics['grade_avg_int'] = grade_avg_int
        
        self.train_df = self._add_feature(
            self.train_df, 'rate_spread',
            self.train_df['int_rate'] - self.train_df['grade'].map(grade_avg_int)
        )
        self.test_df = self._add_feature(
            self.test_df, 'rate_spread',
            self.test_df['int_rate'] - self.test_df['grade'].map(grade_avg_int)
        )
        
        # Revolving balance per account
        self.train_df = self._add_feature(
            self.train_df, 'revol_bal_per_acc',
            self.train_df['revol_bal'] / (self.train_df['total_acc'] + 1)
        )
        self.test_df = self._add_feature(
            self.test_df, 'revol_bal_per_acc',
            self.test_df['revol_bal'] / (self.test_df['total_acc'] + 1)
        )
        
        # Public record presence
        self.train_df = self._add_feature(
            self.train_df, 'has_pub_rec',
            (self.train_df['pub_rec'] > 0).astype(int)
        )
        self.test_df = self._add_feature(
            self.test_df, 'has_pub_rec',
            (self.test_df['pub_rec'] > 0).astype(int)
        )
        
        # Bankruptcy flag
        if 'pub_rec_bankruptcies' in self.train_df.columns:
            self.train_df = self._add_feature(
                self.train_df, 'has_bankruptcy',
                (self.train_df['pub_rec_bankruptcies'] > 0).astype(int)
            )
            self.test_df = self._add_feature(
                self.test_df, 'has_bankruptcy',
                (self.test_df['pub_rec_bankruptcies'] > 0).astype(int)
            )
        
        # Loan amount squared
        self.train_df = self._add_feature(
            self.train_df, 'loan_amnt_sq',
            self.train_df['funded_amnt'] ** 2 / 1e6
        )
        self.test_df = self._add_feature(
            self.test_df, 'loan_amnt_sq',
            self.test_df['funded_amnt'] ** 2 / 1e6
        )
        
        # DTI squared
        self.train_df = self._add_feature(
            self.train_df, 'dti_sq',
            (self.train_df['dti'].fillna(0) ** 2) / 100
        )
        self.test_df = self._add_feature(
            self.test_df, 'dti_sq',
            (self.test_df['dti'].fillna(0) ** 2) / 100
        )
        
        # FICO x Grade interaction
        self.train_df = self._add_feature(
            self.train_df, 'fico_x_grade',
            self.train_df['fico_norm'] * self.train_df['grade_num']
        )
        self.test_df = self._add_feature(
            self.test_df, 'fico_x_grade',
            self.test_df['fico_norm'] * self.test_df['grade_num']
        )
        
        logger.info("Domain-enhanced features applied")
        return self.train_df, self.test_df
    
    def apply_interaction_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply interaction features.
        These are deterministic from existing features.
        """
        logger.info("Applying interaction features...")
        
        # Rate x DTI
        self.train_df = self._add_feature(
            self.train_df, 'rate_x_dti',
            self.train_df['int_rate'] * self.train_df['dti'].fillna(0)
        )
        self.test_df = self._add_feature(
            self.test_df, 'rate_x_dti',
            self.test_df['int_rate'] * self.test_df['dti'].fillna(0)
        )
        
        # FICO x Utilization
        self.train_df = self._add_feature(
            self.train_df, 'fico_x_util',
            self.train_df['fico_norm'] * self.train_df['utilization']
        )
        self.test_df = self._add_feature(
            self.test_df, 'fico_x_util',
            self.test_df['fico_norm'] * self.test_df['utilization']
        )
        
        # Term x Rate
        self.train_df = self._add_feature(
            self.train_df, 'term_x_rate',
            self.train_df['term_60'] * self.train_df['int_rate']
        )
        self.test_df = self._add_feature(
            self.test_df, 'term_x_rate',
            self.test_df['term_60'] * self.test_df['int_rate']
        )
        
        # Loan amount x Rate
        self.train_df = self._add_feature(
            self.train_df, 'loan_x_rate',
            self.train_df['funded_amnt'] * self.train_df['int_rate'] / 1e4
        )
        self.test_df = self._add_feature(
            self.test_df, 'loan_x_rate',
            self.test_df['funded_amnt'] * self.test_df['int_rate'] / 1e4
        )
        
        # DTI x Utilization
        self.train_df = self._add_feature(
            self.train_df, 'dti_x_util',
            self.train_df['dti'].fillna(0) * self.train_df['utilization']
        )
        self.test_df = self._add_feature(
            self.test_df, 'dti_x_util',
            self.test_df['dti'].fillna(0) * self.test_df['utilization']
        )
        
        # Inquiry squared
        self.train_df = self._add_feature(
            self.train_df, 'inq_sq',
            self.train_df['inq_last_6mths'].fillna(0) ** 2
        )
        self.test_df = self._add_feature(
            self.test_df, 'inq_sq',
            self.test_df['inq_last_6mths'].fillna(0) ** 2
        )
        
        # Open accounts squared
        self.train_df = self._add_feature(
            self.train_df, 'open_acc_sq',
            self.train_df['open_acc'].fillna(0) ** 2 / 100
        )
        self.test_df = self._add_feature(
            self.test_df, 'open_acc_sq',
            self.test_df['open_acc'].fillna(0) ** 2 / 100
        )
        
        # Log annual income
        self.train_df = self._add_feature(
            self.train_df, 'log_annual_inc',
            np.log(self.train_df['annual_inc'] + 1)
        )
        self.test_df = self._add_feature(
            self.test_df, 'log_annual_inc',
            np.log(self.test_df['annual_inc'] + 1)
        )
        
        # Log loan amount
        self.train_df = self._add_feature(
            self.train_df, 'log_loan_amnt',
            np.log(self.train_df['funded_amnt'] + 1)
        )
        self.test_df = self._add_feature(
            self.test_df, 'log_loan_amnt',
            np.log(self.test_df['funded_amnt'] + 1)
        )
        
        # Grade one-hot encoding
        grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        for grade in grades:
            self.train_df = self._add_feature(
                self.train_df, f'grade_{grade}',
                (self.train_df['grade'] == grade).astype(int)
            )
            self.test_df = self._add_feature(
                self.test_df, f'grade_{grade}',
                (self.test_df['grade'] == grade).astype(int)
            )
        
        logger.info("Interaction features applied")
        return self.train_df, self.test_df
    
    def apply_ml_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply ML-informed features.
        All aggregates calculated on train only.
        """
        logger.info("Applying ML-informed features...")
        
        # Default rate by purpose (calculated on train)
        if 'purpose' in self.train_df.columns and 'default' in self.train_df.columns:
            purpose_default = self.train_df.groupby('purpose')['default'].mean().to_dict()
            self.statistics['purpose_default'] = purpose_default
            
            self.train_df = self._add_feature(
                self.train_df, 'purpose_default_rate',
                self.train_df['purpose'].map(purpose_default).fillna(0)
            )
            self.test_df = self._add_feature(
                self.test_df, 'purpose_default_rate',
                self.test_df['purpose'].map(purpose_default).fillna(0)
            )
        
        # Default rate by grade (calculated on train)
        if 'grade' in self.train_df.columns and 'default' in self.train_df.columns:
            grade_default = self.train_df.groupby('grade')['default'].mean().to_dict()
            self.statistics['grade_default'] = grade_default
            
            self.train_df = self._add_feature(
                self.train_df, 'grade_default_rate',
                self.train_df['grade'].map(grade_default).fillna(0)
            )
            self.test_df = self._add_feature(
                self.test_df, 'grade_default_rate',
                self.test_df['grade'].map(grade_default).fillna(0)
            )
        
        # Default rate by employment length (calculated on train)
        if 'emp_length' in self.train_df.columns and 'default' in self.train_df.columns:
            emp_default = self.train_df.groupby('emp_length')['default'].mean().to_dict()
            self.statistics['emp_default'] = emp_default
            
            self.train_df = self._add_feature(
                self.train_df, 'emp_length_default_rate',
                self.train_df['emp_length'].map(emp_default).fillna(0)
            )
            self.test_df = self._add_feature(
                self.test_df, 'emp_length_default_rate',
                self.test_df['emp_length'].map(emp_default).fillna(0)
            )
        
        # Default rate by state (calculated on train)
        if 'addr_state' in self.train_df.columns and 'default' in self.train_df.columns:
            state_default = self.train_df.groupby('addr_state')['default'].mean().to_dict()
            self.statistics['state_default'] = state_default
            
            self.train_df = self._add_feature(
                self.train_df, 'state_default_rate',
                self.train_df['addr_state'].map(state_default).fillna(0)
            )
            self.test_df = self._add_feature(
                self.test_df, 'state_default_rate',
                self.test_df['addr_state'].map(state_default).fillna(0)
            )
        
        # Issue month and year
        if 'issue_d' in self.train_df.columns:
            self.train_df = self._add_feature(
                self.train_df, 'issue_month',
                self.train_df['issue_d'].dt.month
            )
            self.test_df = self._add_feature(
                self.test_df, 'issue_month',
                self.test_df['issue_d'].dt.month
            )
            
            self.train_df = self._add_feature(
                self.train_df, 'issue_year',
                self.train_df['issue_d'].dt.year
            )
            self.test_df = self._add_feature(
                self.test_df, 'issue_year',
                self.test_df['issue_d'].dt.year
            )
        
        # Polynomial features
        poly_cols = ['funded_amnt', 'dti', 'int_rate']
        for col in poly_cols:
            if col in self.train_df.columns:
                self.train_df = self._add_feature(
                    self.train_df, f'{col}_cubed',
                    self.train_df[col].fillna(0) ** 3 / 1e9
                )
                self.test_df = self._add_feature(
                    self.test_df, f'{col}_cubed',
                    self.test_df[col].fillna(0) ** 3 / 1e9
                )
        
        logger.info("ML-informed features applied")
        return self.train_df, self.test_df
    
    def get_statistics(self) -> Dict:
        """Return statistics learned from training data"""
        return self.statistics.copy()

def safe_feature_engineering_pipeline(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    plan: str = 'domain_enhanced'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete safe feature engineering pipeline.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        output_dir: Directory to save engineered data
        plan: Feature plan ('baseline_minimal', 'domain_enhanced', 
              'interaction_heavy', 'ml_informed')
    
    Returns:
        Tuple of (train_engineered, test_engineered)
    """
    logger.info(f"Loading train data from {train_path}")
    logger.info(f"Loading test data from {test_path}")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Load original preprocessed data to get profit column
    original_path = train_path.parent / "cleaned_preprocessed.parquet"
    if original_path.exists():
        original_df = pd.read_parquet(original_path)
        
        # ============================================================
        # CRITICAL: DROP LEAKING FEATURES BEFORE MERGING
        # ============================================================
        leaking_patterns = [
            'pymnt', 'rec_', 'recover', 'fee', 'last_', 'next_',
            'total_bal', 'total_bc', 'total_il', 'collection',
            'charge_off', 'installment', 'funded_amnt', 'principal',
            'interest', 'balance'
        ]
        
        # Get columns to drop (excluding 'profit' and 'default')
        columns_to_drop = []
        for col in original_df.columns:
            if col in ['profit', 'default']:
                continue  # Keep these as targets
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in leaking_patterns):
                columns_to_drop.append(col)
                logger.info(f"Dropping leaking feature: {col}")
        
        # Drop leaking columns
        if columns_to_drop:
            original_df = original_df.drop(columns=columns_to_drop, errors='ignore')
            logger.info(f"Dropped {len(columns_to_drop)} leaking features")
        
        # Now safe to merge - only profit column remains from original data
        train_df['profit'] = original_df.loc[train_df.index, 'profit']
        test_df['profit'] = original_df.loc[test_df.index, 'profit']
        logger.info(f"Added profit column to train and test (leaking features removed)")
    else:
        logger.warning("Original preprocessed data not found. Profit column will be missing.")
    
    logger.info(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    engineer = SafeFeatureEngineer(train_df, test_df)
    
    # Apply base features (always)
    train_df, test_df = engineer.apply_base_features()
    
    # Apply additional features based on plan
    if plan in ['domain_enhanced', 'interaction_heavy', 'ml_informed']:
        train_df, test_df = engineer.apply_domain_features()
    
    if plan in ['interaction_heavy', 'ml_informed']:
        train_df, test_df = engineer.apply_interaction_features()
    
    if plan == 'ml_informed':
        train_df, test_df = engineer.apply_ml_features()
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output = output_dir / f"train_engineered_{plan}.parquet"
    test_output = output_dir / f"test_engineered_{plan}.parquet"
    
    train_df.to_parquet(train_output, index=False)
    test_df.to_parquet(test_output, index=False)
    
    logger.info(f"Saved engineered train data to {train_output}")
    logger.info(f"Saved engineered test data to {test_output}")
    
    return train_df, test_df




def main():
    """Main execution"""
    train_path = Path("D:/credit-threshold-profit/data/processed/train_features.parquet")
    test_path = Path("D:/credit-threshold-profit/data/processed/test_features.parquet")
    output_dir = Path("D:/credit-threshold-profit/data/processed")
    
    # Run for all plans
    plans = ['baseline_minimal', 'domain_enhanced', 'interaction_heavy', 'ml_informed']
    
    for plan in plans:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running safe feature engineering for plan: {plan}")
        logger.info(f"{'='*60}")
        
        try:
            safe_feature_engineering_pipeline(
                train_path, test_path, output_dir, plan=plan
            )
        except Exception as e:
            logger.error(f"Failed for plan {plan}: {e}")
    
    logger.info("All safe feature engineering complete")


if __name__ == "__main__":
    main()