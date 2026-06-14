"""
Baseline Data Cleaning for Credit Threshold Profit Project
Minimal cleaning to enable profit-driven threshold modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load Parquet data with error handling"""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def filter_resolved_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with known outcomes (Fully Paid, Charged Off, Default)"""
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df_filtered = df[df['loan_status'].isin(valid_statuses)].copy()
    logger.info(f"Filtered to {len(df_filtered):,} resolved loans")
    logger.info(f"  - Fully Paid: {len(df_filtered[df_filtered['loan_status'] == 'Fully Paid']):,}")
    logger.info(f"  - Charged Off: {len(df_filtered[df_filtered['loan_status'] == 'Charged Off']):,}")
    logger.info(f"  - Default: {len(df_filtered[df_filtered['loan_status'] == 'Default']):,}")
    return df_filtered


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary default flag (1 = default, 0 = fully paid)"""
    df['default'] = df['loan_status'].isin(['Charged Off', 'Default']).astype(int)
    logger.info(f"Binary target created: {df['default'].sum():,} defaults ({df['default'].mean():.2%})")
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with >80% missing or no predictive value"""
    cols_to_drop = [
        # 100% or near-100% missing
        'member_id', 'hardship_flag', 'debt_settlement_flag',
        'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_length',
        'hardship_dpd', 'hardship_loan_status',
        'orig_projected_additional_accrued_interest',
        'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
        'debt_settlement_flag_date', 'settlement_status',
        'settlement_date', 'settlement_amount',
        'settlement_percentage', 'settlement_term',
        
        # Joint application columns (too sparse)
        'annual_inc_joint', 'dti_joint', 'verification_status_joint'
    ]
    
    # Drop sec_app columns (pattern match)
    sec_app_cols = [c for c in df.columns if 'sec_app' in c]
    cols_to_drop.extend(sec_app_cols)
    
    # Useless for modeling
    cols_to_drop.extend(['url', 'desc', 'title', 'zip_code', 'emp_title'])
    
    # Drop columns that exist in dataframe
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df_dropped = df.drop(columns=existing_cols)
    logger.info(f"Dropped {len(existing_cols)} columns, {len(df_dropped.columns)} remaining")
    return df_dropped


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Simple imputation for remaining missing values"""
    # emp_length - mode
    if 'emp_length' in df.columns:
        mode_val = df['emp_length'].mode()[0]
        df['emp_length'] = df['emp_length'].fillna(mode_val)
        logger.info(f"Imputed emp_length with mode: {mode_val}")
    
    # Delinquency columns - 0 means never delinquent
    for col in ['mths_since_last_delinq', 'mths_since_last_record']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            logger.info(f"Imputed {col} with 0")
    
    # revol_util - median
    if 'revol_util' in df.columns:
        median_val = df['revol_util'].median()
        df['revol_util'] = df['revol_util'].fillna(median_val)
        logger.info(f"Imputed revol_util with median: {median_val:.1f}")
    
    # dti - median
    if 'dti' in df.columns:
        median_val = df['dti'].median()
        df['dti'] = df['dti'].fillna(median_val)
        logger.info(f"Imputed dti with median: {median_val:.1f}")
    
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove only obvious data errors"""
    initial_rows = len(df)
    
    # dti > 100 - likely data entry error
    if 'dti' in df.columns:
        df = df[df['dti'] <= 100]
        logger.info(f"Removed {initial_rows - len(df):,} rows with dti > 100")
        initial_rows = len(df)
    
    # annual_inc > 10,000,000 - likely business income
    if 'annual_inc' in df.columns:
        df = df[df['annual_inc'] <= 10_000_000]
        logger.info(f"Removed {initial_rows - len(df):,} rows with annual_inc > 10M")
        initial_rows = len(df)
    
    return df


def drop_correlated_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate correlated columns"""
    cols_to_drop = []
    
    if 'loan_amnt' in df.columns:
        cols_to_drop.append('loan_amnt')
    if 'funded_amnt_inv' in df.columns:
        cols_to_drop.append('funded_amnt_inv')
    if 'fico_range_high' in df.columns:
        cols_to_drop.append('fico_range_high')
    
    df_dropped = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    logger.info(f"Dropped correlated duplicates: {[c for c in cols_to_drop if c in df.columns]}")
    return df_dropped


def save_cleaned_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save cleaned data to Parquet"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path} ({output_path.stat().st_size / 1024**2:.2f} MB)")


def baseline_cleaning_pipeline(
    input_path: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Complete baseline data cleaning pipeline
    """
    logger.info("=" * 60)
    logger.info("STARTING BASELINE DATA CLEANING")
    logger.info("=" * 60)
    
    # Load data
    df = load_data(input_path)
    initial_rows = len(df)
    
    # Apply cleaning steps
    df = filter_resolved_loans(df)
    df = create_binary_target(df)
    df = drop_useless_columns(df)
    df = impute_missing_values(df)
    df = remove_outliers(df)
    df = drop_correlated_duplicates(df)
    
    # Final stats
    logger.info("=" * 60)
    logger.info("CLEANING COMPLETE")
    logger.info(f"Initial rows: {initial_rows:,}")
    logger.info(f"Final rows: {len(df):,}")
    logger.info(f"Rows kept: {len(df) / initial_rows:.1%}")
    logger.info(f"Final columns: {len(df.columns)}")
    logger.info(f"Default rate: {df['default'].mean():.2%}")
    logger.info("=" * 60)
    
    # Save
    save_cleaned_data(df, output_path)
    
    return df


def main():
    """Main execution"""
    input_path = Path(r"D:\credit-threshold-profit\data\raw\accepted_loans.parquet")
    output_path = Path(r"D:\credit-threshold-profit\data\processed\cleaned_baseline.parquet")
    
    df = baseline_cleaning_pipeline(input_path, output_path)
    
    # Quick validation
    logger.info("Quick validation:")
    logger.info(f"  - No missing values: {df.isnull().sum().sum() == 0}")
    logger.info(f"  - Default rate: {df['default'].mean():.2%}")
    logger.info(f"  - Columns: {list(df.columns[:10])}...")


if __name__ == "__main__":
    main()