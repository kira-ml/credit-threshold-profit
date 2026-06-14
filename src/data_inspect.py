import pandas as pd
from pathlib import Path

# Load the cleaned data
df = pd.read_parquet("D:/credit-threshold-profit/data/processed/cleaned_baseline.parquet")

print("="*60)
print("CLEANED DATA INSPECTION")
print("="*60)

# 1. Basic info
print(f"Shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. Check for remaining missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
print(f"\nColumns with missing values: {len(missing)}")
if len(missing) > 0:
    print(missing)

# 3. Check dtypes
print("\nData types (sample):")
print(df.dtypes.head(15))

# 4. Check categorical columns specifically
categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
print(f"\nCategorical columns: {len(categorical_cols)}")
print(categorical_cols[:10])

# 5. Check grade column specifically (since it caused errors)
if 'grade' in df.columns:
    print(f"\nGrade column: {df['grade'].dtype}")
    print(f"Unique values: {df['grade'].unique()}")
    print(f"Null count: {df['grade'].isnull().sum()}")

# 6. Check target distribution
print(f"\nTarget distribution:")
print(df['default'].value_counts(normalize=True))

# 7. Check profit distribution
if 'profit' in df.columns:
    print(f"\nProfit summary:")
    print(df['profit'].describe())
    print(f"Profitable loans: {(df['profit'] > 0).sum():,} ({(df['profit'] > 0).mean():.1%})")
    print(f"Loss-making loans: {(df['profit'] < 0).sum():,} ({(df['profit'] < 0).mean():.1%})")