"""
Data Splitter for Credit Threshold Project
Splits preprocessed data into train/test sets by time (issue_d)
"""

import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(input_path: Path) -> pd.DataFrame:
    """Load preprocessed data"""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Split data by issue_d timestamp.
    
    Args:
        df: DataFrame with 'issue_d' column
        train_ratio: Proportion of data for training (default: 0.8)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Ensure data is sorted by time
    df = df.sort_values('issue_d')
    
    # Verify profit column exists
    if 'profit' not in df.columns:
        raise ValueError("profit column not found in data. Please run preprocessing with profit included.")
    
    # Calculate split index
    split_idx = int(len(df) * train_ratio)
    
    # Split
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Log split info
    logger.info(f"Train: {len(train_df):,} rows ({train_ratio:.1%})")
    logger.info(f"Test: {len(test_df):,} rows ({1-train_ratio:.1%})")
    logger.info(f"Train period: {train_df['issue_d'].min().date()} to {train_df['issue_d'].max().date()}")
    logger.info(f"Test period: {test_df['issue_d'].min().date()} to {test_df['issue_d'].max().date()}")
    
    return train_df, test_df






def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    """Save train and test DataFrames to Parquet"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train_features.parquet"
    test_path = output_dir / "test_features.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    logger.info(f"Saved train data to {train_path}")
    logger.info(f"Saved test data to {test_path}")


def main():
    """Main execution"""
    input_path = Path("D:/credit-threshold-profit/data/processed/cleaned_preprocessed.parquet")
    output_dir = Path("D:/credit-threshold-profit/data/processed")
    
    logger.info("="*60)
    logger.info("DATA SPLITTING")
    logger.info("="*60)
    
    # Load data
    df = load_data(input_path)
    
    # Split by time
    train_df, test_df = split_by_time(df, train_ratio=0.8)
    
    # Save
    save_split(train_df, test_df, output_dir)
    
    logger.info("="*60)
    logger.info("Data splitting complete")
    logger.info("="*60)


if __name__ == "__main__":
    main()