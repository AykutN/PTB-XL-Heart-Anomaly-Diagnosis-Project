"""
Preprocessing Pipeline for PTB-XL+ Dataset

This script performs the complete preprocessing pipeline:
1. Merge PTB-XL metadata with PTB-XL+ features
2. Process labels and create multi-label columns
3. Clean columns (drop sparse/metadata columns)
4. Split data into train/validation/test sets
5. Impute missing values and create missing flags

Usage:
    python preprocessing/pipeline.py
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'preprocessing'))

from merge_and_label import load_and_merge_data, process_labels, analyze_missing_data
import clean_columns
import split_data
import impute_and_flag

def main():
    """Run complete preprocessing pipeline."""
    print("="*80)
    print("PTB-XL+ PREPROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Merge and Label
    print("\n[Step 1/4] Merging datasets and processing labels...")
    df = load_and_merge_data()
    df_processed, dropped_no_label = process_labels(df)
    missing_report = analyze_missing_data(df_processed)
    
    # Save intermediate result
    output_path = os.path.join(BASE_DIR, 'data/processed/ptbxl_merged_labeled.csv')
    print(f"\nSaving merged data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path)
    
    # Step 2: Clean Columns
    print("\n[Step 2/4] Cleaning columns...")
    clean_columns.main()
    
    # Step 3: Split Data
    print("\n[Step 3/4] Splitting data into train/val/test...")
    split_data.main()
    
    # Step 4: Impute Missing Values
    print("\n[Step 4/4] Imputing missing values and creating flags...")
    impute_and_flag.main()
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  - data/processed/train_imputed.csv")
    print("  - data/processed/val_imputed.csv")
    print("  - data/processed/test_imputed.csv")

if __name__ == "__main__":
    main()
