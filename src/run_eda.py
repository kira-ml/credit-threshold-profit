"""
EDA Script for Lending Club Loan Data
Generates comprehensive data quality report and statistical analysis
Outputs: Text report + Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Configuration
INPUT_FILE = Path(r"D:\credit-threshold-profit\data\raw\accepted_loans.parquet")
OUTPUT_DIR = Path(r"D:\credit-threshold-profit\reports\eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(file_path: Path) -> pd.DataFrame:
    """Load parquet data with error handling"""
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def generate_data_quality_report(df: pd.DataFrame, output_file: Path) -> None:
    """Generate comprehensive data quality report"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Dataset overview
        f.write("1. DATASET OVERVIEW\n")
        f.write("-"*40 + "\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB\n\n")
        
        # Column info
        f.write("2. COLUMN INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Column Name':<35} {'Type':<15} {'Non-Null %':<12} {'Unique'}\n")
        f.write("-"*70 + "\n")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_pct = (df[col].notna().sum() / len(df)) * 100
            unique_count = df[col].nunique()
            f.write(f"{col:<35} {dtype:<15} {non_null_pct:<11.1f}% {unique_count:,}\n")
        
        # Missing data summary
        f.write("\n3. MISSING DATA ANALYSIS\n")
        f.write("-"*40 + "\n")
        missing_df = df.isnull().sum().sort_values(ascending=False)
        missing_pct = (missing_df / len(df)) * 100
        
        high_missing = missing_pct[missing_pct > 50]
        med_missing = missing_pct[(missing_pct > 20) & (missing_pct <= 50)]
        low_missing = missing_pct[missing_pct <= 20]
        
        f.write(f"Columns with >50% missing: {len(high_missing)}\n")
        f.write(f"Columns with 20-50% missing: {len(med_missing)}\n")
        f.write(f"Columns with <20% missing: {len(low_missing)}\n\n")
        
        if len(missing_pct[missing_pct > 0]) > 0:
            f.write("Top 10 columns with highest missing rates:\n")
            for col, pct in missing_pct.head(10).items():
                if pct > 0:
                    f.write(f"  • {col}: {pct:.1f}% missing\n")
        
        # Duplicates
        f.write("\n4. DUPLICATE ANALYSIS\n")
        f.write("-"*40 + "\n")
        duplicate_rows = df.duplicated().sum()
        f.write(f"Duplicate rows: {duplicate_rows:,} ({duplicate_rows/len(df)*100:.2f}%)\n")
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        f.write(f"\n5. NUMERIC COLUMNS STATISTICS ({len(numeric_cols)} columns)\n")
        f.write("-"*40 + "\n")
        
        # Summary statistics for numeric columns
        stats = df[numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
        f.write("\nSummary statistics (selected percentiles):\n")
        f.write(stats.round(2).to_string())
        
        # Outlier detection (IQR method)
        f.write("\n\n6. OUTLIER DETECTION (IQR Method)\n")
        f.write("-"*40 + "\n")
        
        outlier_summary = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_pct = (len(outliers) / len(df)) * 100
            
            if outlier_pct > 0:
                outlier_summary.append({
                    'column': col,
                    'outliers_pct': outlier_pct,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
        
        outlier_summary = sorted(outlier_summary, key=lambda x: x['outliers_pct'], reverse=True)
        f.write("Top 10 columns with highest outlier percentages:\n")
        for i, item in enumerate(outlier_summary[:10]):
            f.write(f"  {i+1}. {item['column']}: {item['outliers_pct']:.1f}% outliers\n")
            f.write(f"     Range (IQR): [{item['lower_bound']:.2f}, {item['upper_bound']:.2f}]\n")
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
        f.write(f"\n7. CATEGORICAL COLUMNS ANALYSIS ({len(categorical_cols)} columns)\n")
        f.write("-"*40 + "\n")
        
        for col in categorical_cols[:10]:  # Top 10 only to keep report concise
            value_counts = df[col].value_counts()
            f.write(f"\n{col}:\n")
            f.write(f"  Unique values: {df[col].nunique()}\n")
            f.write(f"  Top 5 categories:\n")
            for val, count in value_counts.head(5).items():
                pct = (count / len(df)) * 100
                f.write(f"    • {val}: {count:,} ({pct:.1f}%)\n")
        
        # Target variable analysis (if loan_status exists)
        if 'loan_status' in df.columns:
            f.write("\n8. TARGET VARIABLE ANALYSIS (loan_status)\n")
            f.write("-"*40 + "\n")
            
            status_counts = df['loan_status'].value_counts()
            f.write("Loan status distribution:\n")
            for status, count in status_counts.items():
                pct = (count / len(df)) * 100
                f.write(f"  • {status}: {count:,} ({pct:.1f}%)\n")
            
            # Check for binary classification potential
            if len(status_counts) == 2:
                f.write("\n→ Dataset is suitable for binary classification\n")
            else:
                f.write(f"\n→ Dataset has {len(status_counts)} classes - consider multi-class or binary mapping\n")

def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate key visualizations for EDA"""
    
    # 1. Missing data heatmap
    plt.figure(figsize=(12, 8))
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_pct) > 0:
        plt.barh(range(len(missing_pct)), missing_pct.values)
        plt.yticks(range(len(missing_pct)), missing_pct.index)
        plt.xlabel('Missing Percentage (%)')
        plt.title('Missing Data by Column')
        plt.tight_layout()
        plt.savefig(output_dir / '1_missing_data.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Numeric columns distributions (top 12 by variance)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols[:12]  # Limit to top 12
    
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            df[col].hist(ax=axes[i], bins=50, edgecolor='black', alpha=0.7)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            # Use ScalarFormatter for scientific notation
            axes[i].xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            axes[i].ticklabel_format(style='scientific', scilimits=(-3, 4), axis='x')
            axes[i].yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            axes[i].ticklabel_format(style='scientific', scilimits=(-3, 4), axis='y')

        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Distribution of Numeric Features', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / '2_numeric_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Correlation heatmap (top 20 correlated features)
    if len(numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Filter to show only correlations > 0.3 or < -0.3
        mask = np.abs(corr_matrix) < 0.3
        corr_matrix_filtered = corr_matrix.copy()
        corr_matrix_filtered[mask] = np.nan
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, 
                   cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / '3_correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Box plots for outlier detection (top 8 numeric columns)
    if len(numeric_cols) >= 4:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:8]):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_ylabel('Value')
            # Use ScalarFormatter for scientific notation
            axes[i].yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            axes[i].ticklabel_format(style='scientific', scilimits=(-3, 4), axis='y')
        
        plt.suptitle('Outlier Detection - Box Plots', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / '4_outlier_boxplots.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Target variable analysis (if loan_status exists)
    if 'loan_status' in df.columns:
        plt.figure(figsize=(10, 6))
        status_counts = df['loan_status'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
               colors=colors[:len(status_counts)], startangle=90)
        plt.title('Loan Status Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / '5_target_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Key features grouped by loan_status
        key_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
        key_features = [f for f in key_features if f in df.columns]
        
        if len(key_features) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(key_features):
                # Group by loan_status (simplify to top 5 statuses)
                top_statuses = df['loan_status'].value_counts().head(5).index
                plot_data = df[df['loan_status'].isin(top_statuses)]
                
                # Calculate means for each status
                means = plot_data.groupby('loan_status')[feature].mean().sort_values()
                means.plot(kind='barh', ax=axes[i], color='skyblue', edgecolor='black')
                axes[i].set_title(f'Average {feature} by Loan Status')
                axes[i].set_xlabel(feature)
            
            plt.suptitle('Key Features by Loan Status', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / '6_features_by_status.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}")

def main():
    """Main execution function"""
    print("="*60)
    print("EDA REPORT GENERATION")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_data(INPUT_FILE)
    
    # Generate text report
    print("\n📝 Generating text report...")
    report_file = OUTPUT_DIR / f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    generate_data_quality_report(df, report_file)
    print(f"✅ Report saved: {report_file}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    create_visualizations(df, OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print(f"📁 Reports saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()