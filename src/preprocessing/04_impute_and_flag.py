import pandas as pd
import numpy as np
import os

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')

def main():
    print("Loading splits...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='ecg_id')
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'), index_col='ecg_id')
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col='ecg_id')
    
    print(f"Train shape: {train_df.shape}")
    
    # Drop heart_axis if present
    if 'heart_axis' in train_df.columns:
        print("Dropping heart_axis column...")
        train_df.drop(columns=['heart_axis'], inplace=True)
        val_df.drop(columns=['heart_axis'], inplace=True)
        test_df.drop(columns=['heart_axis'], inplace=True)
        
    # Re-identify columns with missing values in TRAIN set after dropping
    missing_cols = train_df.columns[train_df.isnull().any()].tolist()
    print(f"\nColumns with missing values in Train: {len(missing_cols)}")
    print(missing_cols)
    
    # Define strategies
    # P-wave features: Missing likely means "No P-wave" -> Fill 0
    p_wave_features = [
        'P_On_Global', 'P_Off_Global', 'P_Dur_Global', 'PR_Int_Global', 
        'HR_Atrial_Global', 'P_AxisFront_Global'
    ]
    
    # Categorical features
    categorical_features = []
    
    for col in missing_cols:
        print(f"Processing {col}...")
        
        # 1. Create Flag Column (is_missing_X)
        # We do this for ALL sets
        train_df[f'is_missing_{col}'] = train_df[col].isnull().astype(int)
        val_df[f'is_missing_{col}'] = val_df[col].isnull().astype(int)
        test_df[f'is_missing_{col}'] = test_df[col].isnull().astype(int)
        
        # 2. Determine Fill Value (FROM TRAIN ONLY)
        if col in categorical_features:
            fill_val = 'MISSING'
        elif any(p in col for p in p_wave_features): # Flexible match
            fill_val = 0.0
        else:
            # Default to median for other numerical features
            # Check if column is numeric first
            if pd.api.types.is_numeric_dtype(train_df[col]):
                fill_val = train_df[col].median()
            else:
                # Fallback for other categoricals if any
                fill_val = train_df[col].mode()[0]
            
        print(f"  -> Filling with: {fill_val}")
        
        # 3. Impute
        train_df[col] = train_df[col].fillna(fill_val)
        val_df[col] = val_df[col].fillna(fill_val)
        test_df[col] = test_df[col].fillna(fill_val)

    # Verify no missing values remain
    total_missing = train_df.isnull().sum().sum() + val_df.isnull().sum().sum() + test_df.isnull().sum().sum()
    if total_missing > 0:
        print(f"\nWARNING: {total_missing} missing values remain!")
        # Check where
        print(train_df.columns[train_df.isnull().any()])
    else:
        print("\nAll missing values imputed successfully.")

    # Save
    print("\nSaving imputed datasets...")
    train_df.to_csv(os.path.join(DATA_DIR, 'train_imputed.csv'))
    val_df.to_csv(os.path.join(DATA_DIR, 'val_imputed.csv'))
    test_df.to_csv(os.path.join(DATA_DIR, 'test_imputed.csv'))
    print("Done.")

if __name__ == "__main__":
    main()
