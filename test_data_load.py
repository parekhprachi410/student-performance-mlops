import pandas as pd
import sys

print("="*60)
print("Testing Data Loading")
print("="*60)

try:
    # Load data
    df = pd.read_csv('data/raw/Student_Performance.csv')
    print(f"✅ Successfully loaded data!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nGrade distribution:")
    print(df['final_grade'].value_counts())
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)
