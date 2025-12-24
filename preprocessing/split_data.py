import pandas as pd
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data/processed/ptbxl_cleaned_columns.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/processed/')

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH, index_col='ecg_id')
    
    if 'strat_fold' not in df.columns:
        raise ValueError("strat_fold column not found!")
        
    print(f"Total records: {len(df)}")
    
    # Split Data
    # Recommended split: 1-8 Train, 9 Val, 10 Test
    train_df = df[df['strat_fold'].isin([1, 2, 3, 4, 5, 6, 7, 8])].copy()
    val_df = df[df['strat_fold'] == 9].copy()
    test_df = df[df['strat_fold'] == 10].copy()
    
    print("\nSplit Sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.2f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)")
    
    # Verify Class Distribution (using one-hot columns)
    diag_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    print("\nClass Distribution (Train):")
    print(train_df[diag_cols].mean() * 100)
    
    print("\nClass Distribution (Val):")
    print(val_df[diag_cols].mean() * 100)
    
    print("\nClass Distribution (Test):")
    print(test_df[diag_cols].mean() * 100)
    
    # Save splits
    print("\nSaving splits...")
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'))
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'))
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'))
    print("Done.")

if __name__ == "__main__":
    main()
