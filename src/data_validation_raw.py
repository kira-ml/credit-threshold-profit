"""
Data Validation Script for Credit Threshold Project
Optimized for LendingClub data with 151 features
Validates raw parquet data integrity, statistical properties, and data quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Production-grade data validator for LendingClub loan data (151 features).
    Performs comprehensive validation with memory-efficient processing.
    """
    
    def __init__(self, file_path: str, chunk_size: int = 100000):
        """
        Initialize validator with file path and chunk size for memory efficiency.
        
        Args:
            file_path: Path to the parquet file
            chunk_size: Number of rows to process at once
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.validation_results = {}
        self.column_stats = {}
        self.quality_issues = []
        
        # Feature categorization based on LendingClub data
        self.feature_categories = self._categorize_features()
        
    def _categorize_features(self) -> Dict[str, List[str]]:
        """
        Categorize the 151 features into groups for targeted validation.
        """
        # All 151 features from your dataset
        all_features = [
            'id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
            'term', 'int_rate', 'installment', 'grade', 'sub_grade',
            'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
            'verification_status', 'issue_d', 'loan_status', 'pymnt_plan',
            'url', 'desc', 'purpose', 'title', 'zip_code', 'addr_state',
            'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low',
            'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
            'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
            'revol_util', 'total_acc', 'initial_list_status', 'out_prncp',
            'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
            'total_rec_int', 'total_rec_late_fee', 'recoveries',
            'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
            'next_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high',
            'last_fico_range_low', 'collections_12_mths_ex_med',
            'mths_since_last_major_derog', 'policy_code', 'application_type',
            'annual_inc_joint', 'dti_joint', 'verification_status_joint',
            'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
            'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
            'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m',
            'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi',
            'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths',
            'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
            'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
            'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
            'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
            'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
            'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
            'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
            'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
            'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
            'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
            'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
            'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
            'revol_bal_joint', 'sec_app_fico_range_low',
            'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
            'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
            'sec_app_revol_util', 'sec_app_open_act_il',
            'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
            'sec_app_collections_12_mths_ex_med',
            'sec_app_mths_since_last_major_derog', 'hardship_flag',
            'hardship_type', 'hardship_reason', 'hardship_status',
            'deferral_term', 'hardship_amount', 'hardship_start_date',
            'hardship_end_date', 'payment_plan_start_date', 'hardship_length',
            'hardship_dpd', 'hardship_loan_status',
            'orig_projected_additional_accrued_interest',
            'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
            'disbursement_method', 'debt_settlement_flag',
            'debt_settlement_flag_date', 'settlement_status',
            'settlement_date', 'settlement_amount', 'settlement_percentage',
            'settlement_term'
        ]
        
        # Categorize features
        categories = {
            'identifiers': ['id', 'member_id', 'url', 'desc', 'title', 'zip_code'],
            'loan_characteristics': [
                'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
                'installment', 'grade', 'sub_grade', 'pymnt_plan', 'purpose',
                'policy_code', 'application_type', 'disbursement_method'
            ],
            'borrower_profile': [
                'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
                'verification_status', 'addr_state', 'annual_inc_joint',
                'dti_joint', 'verification_status_joint'
            ],
            'credit_history': [
                'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
                'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
                'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
                'last_fico_range_high', 'last_fico_range_low',
                'collections_12_mths_ex_med', 'mths_since_last_major_derog',
                'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
                'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
                'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m',
                'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi',
                'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths',
                'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
                'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
                'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
                'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
                'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
                'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
                'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
                'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
                'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
                'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
                'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
                'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit'
            ],
            'co_applicant': [
                'revol_bal_joint', 'sec_app_fico_range_low',
                'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
                'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
                'sec_app_revol_util', 'sec_app_open_act_il',
                'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
                'sec_app_collections_12_mths_ex_med',
                'sec_app_mths_since_last_major_derog'
            ],
            'hardship': [
                'hardship_flag', 'hardship_type', 'hardship_reason',
                'hardship_status', 'deferral_term', 'hardship_amount',
                'hardship_start_date', 'hardship_end_date',
                'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
                'hardship_loan_status',
                'orig_projected_additional_accrued_interest',
                'hardship_payoff_balance_amount', 'hardship_last_payment_amount'
            ],
            'settlement': [
                'debt_settlement_flag', 'debt_settlement_flag_date',
                'settlement_status', 'settlement_date', 'settlement_amount',
                'settlement_percentage', 'settlement_term'
            ],
            'performance': [
                'loan_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
                'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
                'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d',
                'last_credit_pull_d', 'initial_list_status'
            ],
            'dates': [
                'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d',
                'last_credit_pull_d', 'hardship_start_date', 'hardship_end_date',
                'payment_plan_start_date', 'debt_settlement_flag_date',
                'settlement_date', 'sec_app_earliest_cr_line'
            ]
        }
        
        return categories
    
    def validate_file_integrity(self) -> bool:
        """Validate parquet file integrity"""
        logger.info("=== Validating File Integrity ===")
        
        try:
            if not self.file_path.exists():
                logger.error(f"File not found: {self.file_path}")
                return False
            
            file_size_mb = self.file_path.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            # Read first chunk to verify
            df = pd.read_parquet(self.file_path)
            
            if df.empty:
                logger.error("DataFrame is empty")
                return False
            
            # Verify 151 features
            actual_features = len(df.columns)
            if actual_features != 151:
                logger.warning(f"Expected 151 features, found {actual_features}")
            
            logger.info(f"File validation passed: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return True
            
        except Exception as e:
            logger.error(f"File integrity check failed: {e}")
            return False
    
    def validate_feature_presence(self) -> Tuple[List[str], List[str]]:
        """Validate all 151 features are present"""
        logger.info("=== Validating Feature Presence ===")
        
        all_features = self.feature_categories['identifiers'] + \
                      self.feature_categories['loan_characteristics'] + \
                      self.feature_categories['borrower_profile'] + \
                      self.feature_categories['credit_history'] + \
                      self.feature_categories['co_applicant'] + \
                      self.feature_categories['hardship'] + \
                      self.feature_categories['settlement'] + \
                      self.feature_categories['performance']
        
        # Remove duplicates
        all_features = list(dict.fromkeys(all_features))
        
        df = pd.read_parquet(self.file_path)
        actual_features = df.columns.tolist()
        
        missing = [f for f in all_features if f not in actual_features]
        extra = [f for f in actual_features if f not in all_features]
        
        if missing:
            logger.warning(f"Missing features: {missing}")
        if extra:
            logger.info(f"Extra features: {extra}")
        
        return missing, extra
    
    def validate_numeric_features(self) -> Dict:
        """Validate numeric features for statistical properties"""
        logger.info("=== Validating Numeric Features ===")
        
        df = pd.read_parquet(self.file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {}
        for col in numeric_cols:
            if df[col].isna().all():
                continue
                
            col_stats = {
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'median': float(df[col].median()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing_pct': round((df[col].isna().sum() / len(df)) * 100, 2),
                'zeros_pct': round((df[col] == 0).sum() / len(df) * 100, 2) if col != 'id' else 0
            }
            
            # Check for outliers (IQR method)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
            col_stats['outliers_pct'] = round((outliers / len(df)) * 100, 2)
            
            stats[col] = col_stats
        
        # Log problematic features
        high_missing = [col for col, s in stats.items() if s['missing_pct'] > 30]
        if high_missing:
            logger.warning(f"High missing values (>30%): {high_missing[:10]}...")
        
        high_outliers = [col for col, s in stats.items() if s['outliers_pct'] > 5]
        if high_outliers:
            logger.warning(f"High outliers (>5%): {high_outliers[:10]}...")
        
        return stats
    
    def validate_categorical_features(self) -> Dict:
        """Validate categorical features"""
        logger.info("=== Validating Categorical Features ===")
        
        df = pd.read_parquet(self.file_path)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        cat_stats = {}
        for col in cat_cols:
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)
            
            cat_stats[col] = {
                'unique_values': unique_count,
                'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                'mode_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'high_cardinality': unique_count > 100,
                'missing_pct': round((df[col].isna().sum() / len(df)) * 100, 2)
            }
            
            if unique_count > 100:
                logger.warning(f"High cardinality in {col}: {unique_count} unique values")
        
        return cat_stats
    
    def validate_date_features(self) -> Dict:
        """Validate date features"""
        logger.info("=== Validating Date Features ===")
        
        date_cols = [
            'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d',
            'last_credit_pull_d', 'hardship_start_date', 'hardship_end_date',
            'payment_plan_start_date', 'debt_settlement_flag_date',
            'settlement_date', 'sec_app_earliest_cr_line'
        ]
        
        df = pd.read_parquet(self.file_path)
        date_stats = {}
        
        for col in date_cols:
            if col not in df.columns:
                continue
                
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                invalid = parsed.isna().sum()
                
                date_stats[col] = {
                    'invalid_pct': round((invalid / len(df)) * 100, 2),
                    'min': parsed.min().strftime('%Y-%m-%d') if not pd.isna(parsed.min()) else None,
                    'max': parsed.max().strftime('%Y-%m-%d') if not pd.isna(parsed.max()) else None,
                    'range_days': (parsed.max() - parsed.min()).days if not pd.isna(parsed.min()) else None
                }
                
                if invalid > 0:
                    logger.warning(f"Invalid dates in {col}: {date_stats[col]['invalid_pct']}%")
                    
            except Exception as e:
                logger.warning(f"Could not parse {col} as date: {e}")
                date_stats[col] = {'error': str(e)}
        
        return date_stats
    
    def generate_report(self, output_path: str = None) -> Dict:
        """Generate comprehensive validation report"""
        logger.info("="*60)
        logger.info("GENERATING VALIDATION REPORT")
        logger.info("="*60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'file_path': str(self.file_path),
            'file_integrity': self.validate_file_integrity(),
            'feature_presence': self.validate_feature_presence(),
            'numeric_stats': self.validate_numeric_features(),
            'categorical_stats': self.validate_categorical_features(),
            'date_stats': self.validate_date_features()
        }
        
        # Summary
        numeric_count = len(results['numeric_stats'])
        categorical_count = len(results['categorical_stats'])
        date_count = len(results['date_stats'])
        
        summary = {
            'total_features': 151,
            'numeric_features': numeric_count,
            'categorical_features': categorical_count,
            'date_features': date_count,
            'missing_features': len(results['feature_presence'][0]),
            'extra_features': len(results['feature_presence'][1])
        }
        
        results['summary'] = summary
        
        # Save report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"File: {self.file_path}")
        print(f"Features: {summary['total_features']}")
        print(f"Numeric: {summary['numeric_features']}")
        print(f"Categorical: {summary['categorical_features']}")
        print(f"Date: {summary['date_features']}")
        print(f"Missing features: {summary['missing_features']}")
        print(f"Extra features: {summary['extra_features']}")
        print("="*60)
        
        return results
    
    def quick_check(self) -> bool:
        """Quick validation for CI/CD"""
        logger.info("Running quick validation...")
        
        try:
            if not self.validate_file_integrity():
                return False
            
            missing, extra = self.validate_feature_presence()
            if missing:
                logger.error(f"Missing required features: {missing}")
                return False
            
            logger.info("Quick validation passed!")
            return True
            
        except Exception as e:
            logger.error(f"Quick validation failed: {e}")
            return False


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate LendingClub loan data')
    parser.add_argument('--file', type=str, default='data/raw/accepted_loans.parquet',
                       help='Path to parquet file')
    parser.add_argument('--output', type=str, default='reports/data_validation/validation_report.json',
                       help='Output path for report')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation only')
    
    args = parser.parse_args()
    
    validator = DataValidator(args.file)
    
    if args.quick:
        success = validator.quick_check()
        exit(0 if success else 1)
    else:
        validator.generate_report(args.output)


if __name__ == "__main__":
    main()