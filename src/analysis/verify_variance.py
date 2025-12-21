import pandas as pd
import numpy as np
import os

BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
INPUT_PATH = os.path.join(BASE_DIR, 'data/processed/train_imputed.csv')

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH, index_col='ecg_id')
    
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    drop_cols = target_cols + ['strat_fold', 'scp_codes', 'diagnostic_superclass']
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    
    print(f"Total Numeric Features: {X.shape[1]}")
    
    variances = X.var()
    
    count_below_0_05 = (variances < 0.05).sum()
    count_below_0_02 = (variances < 0.02).sum()
    
    print(f"Features with variance < 0.05: {count_below_0_05}")
    print(f"Features with variance < 0.02: {count_below_0_02}")
    
    # Check specific low variance features
    print("\nSample low variance features:")
    print(variances[variances < 0.05].head(10))

if __name__ == "__main__":
    main()
