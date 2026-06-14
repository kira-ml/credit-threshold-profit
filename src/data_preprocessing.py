"""
Data Preprocessing for Feature Engineering Preparation
Runs on cleaned baseline data before feature engineering
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


def load_data(input_path: Path) -> pd.DataFrame:
    """Load cleaned baseline data"""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Drop columns with missing percentage above threshold.
    
    Args:
        df: Input DataFrame
        threshold: Drop columns with missing % > threshold (default: 90%)
    
    Returns:
        DataFrame with sparse columns removed
    """
    missing_pct = df.isnull().mean()
    
    # Exclude 'profit' from being dropped
    if 'profit' in df.columns:
        missing_pct = missing_pct.drop('profit', errors='ignore')
    
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold:.0%} missing")
    else:
        logger.info("No columns with >{threshold:.0%} missing")
    
    return df





def convert_categorical_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all categorical columns to string dtype.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with categorical columns as strings
    """
    cat_cols = df.select_dtypes(include=['category']).columns
    if len(cat_cols) > 0:
        df = df.copy()
        for col in cat_cols:
            df[col] = df[col].astype(str)
        logger.info(f"Converted {len(cat_cols)} categorical columns to string")
    return df


def create_grade_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create grade_num feature from grade column.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with grade_num column
    """
    if 'grade' not in df.columns:
        logger.warning("grade column not found, skipping grade_num creation")
        return df
    
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df = df.copy()
    df['grade_num'] = df['grade'].map(grade_map)
    
    # Fill unmapped values with 0 (safety)
    df['grade_num'] = df['grade_num'].fillna(0)
    
    logger.info("Created grade_num feature")
    return df




def create_emp_length_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create emp_length_num feature from emp_length column.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with emp_length_num column
    """
    if 'emp_length' not in df.columns:
        logger.warning("emp_length column not found, skipping emp_length_num creation")
        return df
    
    emp_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10
    }
    df = df.copy()
    df['emp_length_num'] = df['emp_length'].map(emp_map).fillna(0)
    
    logger.info("Created emp_length_num feature")
    return df



def impute_low_missing(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Impute columns with missing percentage below threshold.
    
    Args:
        df: Input DataFrame
        threshold: Impute columns with missing % < threshold (default: 5%)
    
    Returns:
        DataFrame with imputed values
    """
    missing_pct = df.isnull().mean()
    cols_to_impute = missing_pct[(0 < missing_pct) & (missing_pct < threshold)].index.tolist()
    
    if not cols_to_impute:
        logger.info("No columns with <{threshold:.0%} missing to impute")
        return df
    
    df = df.copy()
    imputed_count = 0
    
    for col in cols_to_impute:
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            # Mode imputation for categorical
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                imputed_count += 1
        else:
            # Median imputation for numeric
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_count += 1
    
    logger.info(f"Imputed {imputed_count} columns with <{threshold:.0%} missing")
    return df


def remove_zero_funded_loans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove loans with funded_amnt <= 0.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with zero-funded loans removed
    """
    if 'funded_amnt' not in df.columns:
        logger.warning("funded_amnt column not found, skipping removal")
        return df
    
    initial_rows = len(df)
    df = df[df['funded_amnt'] > 0]
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows:,} rows with funded_amnt <= 0")
    
    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save preprocessed data to Parquet"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved preprocessed data to {output_path}")
    logger.info(f"Final shape: {df.shape}")



def preprocess_for_features(
    input_path: Path,
    output_path: Path,
    sparse_threshold: float = 0.90,
    missing_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for feature engineering.
    
    Args:
        input_path: Path to cleaned baseline data
        output_path: Path to save preprocessed data
        sparse_threshold: Drop columns with missing % above this threshold
        missing_threshold: Impute columns with missing % below this threshold
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info("="*60)
    logger.info("STARTING PREPROCESSING FOR FEATURE ENGINEERING")
    logger.info("="*60)
    
    # Load data
    df = load_data(input_path)
    initial_shape = df.shape
    
    # Verify profit column exists
    if 'profit' not in df.columns:
        logger.warning("profit column not found in input data - this may cause issues in feature engineering")
    else:
        logger.info(f"profit column present with {df['profit'].notna().sum():,} non-null values")
    
    # Apply preprocessing steps
    df = drop_sparse_columns(df, threshold=sparse_threshold)
    df = convert_categorical_to_string(df)
    df = create_grade_numeric(df)
    df = create_emp_length_numeric(df)
    df = impute_low_missing(df, threshold=missing_threshold)
    df = remove_zero_funded_loans(df)
    
    # Log results
    logger.info("="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"Initial shape: {initial_shape}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Rows kept: {len(df) / initial_shape[0]:.1%}")
    logger.info("="*60)
    
    # Save
    save_data(df, output_path)
    
    return df




def main():
    """Main execution"""
    input_path = Path("D:/credit-threshold-profit/data/processed/cleaned_baseline.parquet")
    output_path = Path("D:/credit-threshold-profit/data/processed/cleaned_preprocessed.parquet")
    
    df = preprocess_for_features(input_path, output_path)
    
    # Quick validation
    logger.info("\nQuick validation:")
    logger.info(f"  - No missing values: {df.isnull().sum().sum() == 0}")
    if 'grade_num' in df.columns:
        logger.info(f"  - grade_num created: {df['grade_num'].dtype}")
    logger.info(f"  - Rows: {len(df):,}")
    logger.info(f"  - Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()