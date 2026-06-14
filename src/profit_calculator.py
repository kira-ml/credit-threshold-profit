"""
Profit Calculation Module for Credit Threshold Project
Configurable: Switch between Baseline and Advanced profit calculations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Optional, Union, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProfitMethod(Enum):
    """Profit calculation methods available"""
    BASELINE = "baseline"
    ADVANCED = "advanced"
    HYBRID = "hybrid"  # Use advanced if columns exist, fallback to baseline


class ProfitCalculator:
    """
    Configurable profit calculator for LendingClub loans.
    
    Methods:
        - Baseline: profit = total_pymnt - funded_amnt
        - Advanced: profit = (interest + principal_recovered + late_fees + recoveries) 
                           - (funded_amnt + collection_fees)
        - Hybrid: Use advanced if columns exist, else baseline
    
    Example:
        calculator = ProfitCalculator(method=ProfitMethod.ADVANCED)
        df['profit'] = df.apply(calculator.calculate, axis=1)
    """
    
    def __init__(
        self,
        method: Union[ProfitMethod, str] = ProfitMethod.BASELINE,
        fallback_to_baseline: bool = True,
        verbose: bool = False
    ):
        """
        Initialize profit calculator with configuration.
        
        Args:
            method: ProfitMethod enum or string ('baseline', 'advanced', 'hybrid')
            fallback_to_baseline: If True, use baseline when advanced columns missing
            verbose: If True, log warnings and info messages
        """
        self.method = method if isinstance(method, ProfitMethod) else ProfitMethod(method.lower())
        self.fallback_to_baseline = fallback_to_baseline
        self.verbose = verbose
        
        # Column requirements for each method
        self.required_columns = {
            ProfitMethod.BASELINE: ['total_pymnt', 'funded_amnt'],
            ProfitMethod.ADVANCED: [
                'total_rec_int', 'total_rec_prncp', 'total_rec_late_fee',
                'recoveries', 'collection_recovery_fee', 'funded_amnt'
            ]
        }
        
        if self.verbose:
            logger.info(f"ProfitCalculator initialized with method: {self.method.value}")
    
    def _validate_columns(self, row: pd.Series, method: ProfitMethod) -> bool:
        """Check if required columns exist in row"""
        required = self.required_columns[method]
        missing = [col for col in required if col not in row.index]
        
        if missing and self.verbose:
            logger.warning(f"Missing columns for {method.value}: {missing}")
        
        return len(missing) == 0
    
    def _calculate_baseline(self, row: pd.Series) -> float:
        """
        Baseline profit: total payments received minus funded amount.
        
        profit = total_pymnt - funded_amnt
        """
        if not self._validate_columns(row, ProfitMethod.BASELINE):
            return np.nan
        
        return row['total_pymnt'] - row['funded_amnt']
    
    def _calculate_advanced(self, row: pd.Series) -> float:
        """
        Advanced profit: detailed cash flow components.
        
        profit = (interest + principal_recovered + late_fees + recoveries)
                - (funded_amnt + collection_fees)
        """
        if not self._validate_columns(row, ProfitMethod.ADVANCED):
            if self.fallback_to_baseline:
                if self.verbose:
                    logger.debug("Advanced method missing columns, falling back to baseline")
                return self._calculate_baseline(row)
            return np.nan
        
        interest = row['total_rec_int']
        principal_recovered = row['total_rec_prncp']
        late_fees = row['total_rec_late_fee']
        recoveries = row['recoveries']
        collection_fees = row['collection_recovery_fee']
        funded = row['funded_amnt']
        
        inflows = interest + principal_recovered + late_fees + recoveries
        outflows = funded + collection_fees
        
        return inflows - outflows
    
    def calculate(self, row: pd.Series) -> float:
        """
        Calculate profit based on configured method.
        
        Args:
            row: Pandas Series representing a single loan
            
        Returns:
            Profit amount (float)
        """
        if self.method == ProfitMethod.BASELINE:
            return self._calculate_baseline(row)
        
        elif self.method == ProfitMethod.ADVANCED:
            return self._calculate_advanced(row)
        
        elif self.method == ProfitMethod.HYBRID:
            # Try advanced first, fallback to baseline if columns missing
            if self._validate_columns(row, ProfitMethod.ADVANCED):
                return self._calculate_advanced(row)
            else:
                if self.verbose:
                    logger.debug("Hybrid: Advanced columns missing, using baseline")
                return self._calculate_baseline(row)
        
        else:
            raise ValueError(f"Unknown profit method: {self.method}")
    
    def calculate_batch(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate profit for entire DataFrame efficiently.
        
        Args:
            df: DataFrame with required columns
            
        Returns:
            Series of profit values
        """
        # Check if DataFrame has required columns at batch level
        if self.method == ProfitMethod.BASELINE:
            return df['total_pymnt'] - df['funded_amnt']
        
        elif self.method == ProfitMethod.ADVANCED:
            # Check if all advanced columns exist
            required = self.required_columns[ProfitMethod.ADVANCED]
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                if self.fallback_to_baseline:
                    if self.verbose:
                        logger.warning(f"Advanced batch: missing {missing}, using baseline")
                    return df['total_pymnt'] - df['funded_amnt']
                else:
                    raise ValueError(f"Missing columns for advanced calculation: {missing}")
            
            inflows = (
                df['total_rec_int'] + 
                df['total_rec_prncp'] + 
                df['total_rec_late_fee'] + 
                df['recoveries']
            )
            outflows = df['funded_amnt'] + df['collection_recovery_fee']
            return inflows - outflows
        
        elif self.method == ProfitMethod.HYBRID:
            # Try advanced, fallback to baseline
            required = self.required_columns[ProfitMethod.ADVANCED]
            missing = [c for c in required if c not in df.columns]
            
            if not missing:
                inflows = (
                    df['total_rec_int'] + 
                    df['total_rec_prncp'] + 
                    df['total_rec_late_fee'] + 
                    df['recoveries']
                )
                outflows = df['funded_amnt'] + df['collection_recovery_fee']
                return inflows - outflows
            else:
                if self.verbose:
                    logger.debug(f"Hybrid batch: missing {missing}, using baseline")
                return df['total_pymnt'] - df['funded_amnt']
        
        else:
            raise ValueError(f"Unknown profit method: {self.method}")


def create_profit_config(
    method: str = "baseline",
    fallback_to_baseline: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Create configuration dictionary for profit calculation.
    
    Args:
        method: 'baseline', 'advanced', or 'hybrid'
        fallback_to_baseline: Fall back to baseline if advanced columns missing
        verbose: Log warnings
        
    Returns:
        Configuration dictionary
    """
    return {
        "method": method,
        "fallback_to_baseline": fallback_to_baseline,
        "verbose": verbose
    }


def calculate_profits(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Main function to calculate profits on a DataFrame.
    
    Args:
        df: DataFrame with loan data
        config: Configuration dict (if None, uses baseline)
        
    Returns:
        DataFrame with profit column added
    """
    if config is None:
        config = create_profit_config(method="baseline")
    
    calculator = ProfitCalculator(
        method=config.get("method", "baseline"),
        fallback_to_baseline=config.get("fallback_to_baseline", True),
        verbose=config.get("verbose", False)
    )
    
    logger.info(f"Calculating profits using {config['method']} method")
    
    # Use batch calculation for efficiency
    df = df.copy()
    df['profit'] = calculator.calculate_batch(df)
    
    # Summary statistics
    profit_positive = (df['profit'] > 0).sum()
    profit_negative = (df['profit'] < 0).sum()
    
    logger.info(f"Profit calculation complete:")
    logger.info(f"  - Total loans: {len(df):,}")
    logger.info(f"  - Profitable loans: {profit_positive:,} ({profit_positive/len(df):.1%})")
    logger.info(f"  - Loss-making loans: {profit_negative:,} ({profit_negative/len(df):.1%})")
    logger.info(f"  - Mean profit: ${df['profit'].mean():.2f}")
    logger.info(f"  - Median profit: ${df['profit'].median():.2f}")
    logger.info(f"  - Total portfolio profit: ${df['profit'].sum():,.2f}")
    
    return df


def main():
    """Main execution with profit merging"""
    # Load cleaned data
    input_path = Path(r"D:\credit-threshold-profit\data\processed\cleaned_baseline.parquet")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from cleaned baseline")
    
    # Calculate profits (baseline method)
    config_baseline = create_profit_config(method="baseline", verbose=True)
    df_profit = calculate_profits(df, config_baseline)
    
    # MERGE PROFIT BACK INTO CLEANED BASELINE
    logger.info("Merging profit column into cleaned baseline...")
    df['profit'] = df_profit['profit']
    df.to_parquet(input_path, index=False)
    logger.info(f"✅ Profit column added to {input_path}")
    
    # Also save profit-only file for reference
    output_dir = Path(r"D:\credit-threshold-profit\data\processed")
    df_profit.to_parquet(output_dir / "profits_baseline.parquet", index=False)
    
    logger.info("Profit calculation complete and merged to cleaned baseline")


if __name__ == "__main__":
    main()




    