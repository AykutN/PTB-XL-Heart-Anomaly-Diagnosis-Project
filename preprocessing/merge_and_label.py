import pandas as pd
import ast
import os
import numpy as np

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTBXL_DB_PATH = os.path.join(BASE_DIR, 'ptb-xl/ptbxl_database.csv')
FEATURES_PATH = os.path.join(BASE_DIR, 'ptb-xl+/features/12sl_features.csv')
SCP_STATEMENTS_PATH = os.path.join(BASE_DIR, 'ptb-xl/scp_statements.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data/processed/ptbxl_merged_labeled.csv')

def load_and_merge_data():
    print("Loading datasets...")
    # Load Database
    df = pd.read_csv(PTBXL_DB_PATH, index_col='ecg_id')
    print(f"PTB-XL Database shape: {df.shape}")
    
    # Load Features
    features_df = pd.read_csv(FEATURES_PATH, index_col='ecg_id')
    print(f"12SL Features shape: {features_df.shape}")
    
    # Merge
    # We use inner join to ensure we have both metadata and features for the records
    merged_df = df.join(features_df, how='inner')
    print(f"Merged DataFrame shape: {merged_df.shape}")
    
    return merged_df

def process_labels(df):
    print("\nProcessing labels...")
    # Load SCP statements
    scp_df = pd.read_csv(SCP_STATEMENTS_PATH, index_col=0)
    
    # Create mapping: code -> diagnostic_class
    # We only care about codes that have a diagnostic_class
    code_to_class = scp_df[scp_df['diagnostic_class'].notna()]['diagnostic_class'].to_dict()
    
    # Parse scp_codes
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    
    # Function to map codes to superclasses
    def get_superclasses(codes_dict):
        superclasses = set()
        for code in codes_dict.keys():
            if code in code_to_class:
                superclasses.add(code_to_class[code])
        return list(superclasses)
    
    df['diagnostic_superclass'] = df['scp_codes'].apply(get_superclasses)
    
    # Filter rows with no superclass
    initial_count = len(df)
    df_filtered = df[df['diagnostic_superclass'].map(len) > 0].copy()
    dropped_count = initial_count - len(df_filtered)
    
    print(f"Dropped {dropped_count} records with no valid diagnostic superclass.")
    print(f"Remaining records: {len(df_filtered)}")
    
    # Create Multi-label columns (One-Hot Encoding for superclasses)
    # The 5 superclasses are: NORM, MI, STTC, CD, HYP
    all_superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    for sc in all_superclasses:
        df_filtered[sc] = df_filtered['diagnostic_superclass'].apply(lambda x: 1 if sc in x else 0)
        
    return df_filtered, dropped_count

def analyze_missing_data(df):
    print("\nAnalyzing missing data...")
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    
    total_rows = len(df)
    missing_report = []
    
    if not missing_counts.empty:
        print("Columns with missing values:")
        for col, count in missing_counts.items():
            ratio = (count / total_rows) * 100
            print(f"  - {col}: {count} missing ({ratio:.2f}%)")
            missing_report.append({'column': col, 'missing_count': count, 'missing_ratio': ratio})
    else:
        print("No missing values found in the merged dataframe.")
        
    return missing_report

def main():
    # 1. Load and Merge
    df = load_and_merge_data()
    
    # 2. Process Labels
    df_processed, dropped_no_label = process_labels(df)
    
    # 3. Analyze Missing Data
    missing_report = analyze_missing_data(df_processed)
    
    # 4. Save
    print(f"\nSaving processed data to {OUTPUT_PATH}...")
    df_processed.to_csv(OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
