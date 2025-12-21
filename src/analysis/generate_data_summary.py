import pandas as pd
import numpy as np
import io
import os

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
INPUT_PATH = os.path.join(BASE_DIR, 'data/processed/ptbxl_cleaned_columns.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports/data_summary')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading data...")
    # Using the cleaned dataset before imputation to show original missing values
    df = pd.read_csv(INPUT_PATH, index_col='ecg_id')
    
    # 1. .info() Output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    with open(os.path.join(OUTPUT_DIR, 'dataset_info.txt'), 'w') as f:
        f.write(info_str)
    print("Saved .info() output.")
    
    # 2. .describe() Output (First 10 columns for brevity)
    # Transpose for better readability in report
    describe_df = df.describe().transpose().head(10)
    describe_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_describe_head.csv'))
    print("Saved .describe() output (head).")
    
    # 3. isnull().sum() Output (Top 20 missing columns)
    missing_df = df.isnull().sum().sort_values(ascending=False).head(20)
    missing_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_missing_counts.csv'), header=['missing_count'])
    print("Saved isnull().sum() output.")

    # Print to console for immediate copy-paste
    print("\n" + "="*50)
    print("1. .info() ÇIKTISI")
    print("="*50)
    print(info_str)
    
    print("\n" + "="*50)
    print("2. .describe().head(10) ÇIKTISI")
    print("="*50)
    print(describe_df.to_string())
    
    print("\n" + "="*50)
    print("3. isnull().sum().head(20) ÇIKTISI")
    print("="*50)
    print(missing_df.to_string())

if __name__ == "__main__":
    main()
