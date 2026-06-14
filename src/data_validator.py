"""
Data Validation for Leakage Prevention
Checks for common data leakage issues before training
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


class DataValidator:
    """
    Validates data for leakage issues before training.
    Checks:
        - Future-dependent columns (leakage)
        - Time ordering for train/test split
        - Date column presence
        - Aggregate feature leakage
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
        self.passed = True
        
        # Columns that leak future information
        self.future_dependent_cols = [
            'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
            'total_rec_int', 'total_rec_late_fee', 'recoveries',
            'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
            'next_pymnt_d', 'out_prncp', 'out_prncp_inv',
            'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low'
        ]
    
    def check_future_columns(self) -> None:
        """Check if any future-dependent columns are still present"""
        present = [col for col in self.future_dependent_cols if col in self.df.columns]
        
        if present:
            self.issues.append(f"Future-dependent columns present: {present[:5]}... ({len(present)} total)")
            self.passed = False
            logger.warning(f"Found {len(present)} future-dependent columns")
        else:
            logger.info("No future-dependent columns found")
    
    def check_time_column(self) -> None:
        """Check if issue_d exists and is datetime"""
        if 'issue_d' not in self.df.columns:
            self.issues.append("Missing 'issue_d' column for time-based split")
            self.passed = False
            return
        
        if not pd.api.types.is_datetime64_any_dtype(self.df['issue_d']):
            self.issues.append("'issue_d' is not datetime type - convert with pd.to_datetime()")
            self.passed = False
        else:
            logger.info("'issue_d' column exists and is datetime type")
    
    def check_aggregate_leakage(self) -> None:
        """Check for aggregate features that might leak"""
        # Common aggregate feature names
        aggregate_features = ['_default_rate', '_mean', '_avg', '_count']
        
        for col in self.df.columns:
            if any(feat in col for feat in aggregate_features):
                # If aggregate feature exists, warn but don't fail
                logger.warning(f"Potential aggregate feature found: {col}")
                logger.info("Make sure this was calculated using train data only")
    
    def check_time_range(self) -> None:
        """Print time range info for reference"""
        if 'issue_d' in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df['issue_d']):
            min_date = self.df['issue_d'].min()
            max_date = self.df['issue_d'].max()
            logger.info(f"Time range: {min_date.date()} to {max_date.date()}")
            logger.info(f"Total time span: {(max_date - min_date).days} days")
    
    def validate(self) -> bool:
        """
        Run all validation checks.
        
        Returns:
            True if no critical issues found, False otherwise
        """
        logger.info("="*60)
        logger.info("DATA VALIDATION FOR LEAKAGE PREVENTION")
        logger.info("="*60)
        
        self.issues = []
        self.passed = True
        
        # Run checks
        self.check_future_columns()
        self.check_time_column()
        self.check_aggregate_leakage()
        self.check_time_range()
        
        # Summary
        logger.info("="*60)
        if self.passed:
            logger.info("✅ Validation passed: No critical leakage issues detected")
        else:
            logger.warning("❌ Validation failed: Issues detected")
            for issue in self.issues:
                logger.warning(f"  - {issue}")
            logger.info("Fix these issues before proceeding to training")
        
        logger.info("="*60)
        return self.passed
    
    def get_issues(self) -> list:
        """Return list of detected issues"""
        return self.issues.copy()


def validate_data(file_path: Path) -> bool:
    """
    Convenience function to validate a data file.
    
    Args:
        file_path: Path to Parquet file
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info(f"Validating: {file_path}")
    df = pd.read_parquet(file_path)
    validator = DataValidator(df)
    return validator.validate()


def main():
    """Main execution"""
    input_path = Path("D:\credit-threshold-profit\data\processed\cleaned_baseline.parquet")
    
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return
    
    # Run validation
    passed = validate_data(input_path)
    
    if passed:
        logger.info("\n✅ Data is ready for training")
        logger.info("Next steps:")
        logger.info("  1. Drop future-dependent columns if present")
        logger.info("  2. Split by time (issue_d)")
        logger.info("  3. Train model on train split only")
    else:
        logger.info("\n❌ Fix issues before proceeding to training")


if __name__ == "__main__":
    main()