import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/Machine Learning IU/'
INPUT_PATH = os.path.join(BASE_DIR, 'data/processed/ptbxl_cleaned_columns.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports/missing_data_analysis')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH, index_col='ecg_id')
    
    # 1. Identify Missing Columns
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if missing_counts.empty:
        print("No missing data found.")
        return

    print("\nMissing Data Counts:")
    print(missing_counts)
    
    # 2. Visualization: Bar Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_counts.index, y=missing_counts.values / len(df) * 100)
    plt.xticks(rotation=45, ha='right')
    plt.title('Percentage of Missing Values by Feature')
    plt.ylabel('Missing %')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values_barplot.png'))
    print(f"\nSaved bar plot to {os.path.join(OUTPUT_DIR, 'missing_values_barplot.png')}")
    
    # 3. Correlation with Diagnostics
    # Create binary missing flags
    missing_cols = missing_counts.index.tolist()
    for col in missing_cols:
        df[f'missing_{col}'] = df[col].isnull().astype(int)
        
    # Diagnostic columns
    diag_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    print("\nCorrelation Analysis (Missingness vs Diagnosis):")
    print("-" * 60)
    
    results = []
    
    for feature in missing_cols:
        missing_flag = f'missing_{feature}'
        
        # Subset where feature is missing
        missing_subset = df[df[missing_flag] == 1]
        present_subset = df[df[missing_flag] == 0]
        
        row = {'Feature': feature, 'Missing_Count': len(missing_subset)}
        
        for diag in diag_cols:
            # Calculate prevalence of diagnosis in missing vs present groups
            prob_missing = missing_subset[diag].mean() * 100
            prob_present = present_subset[diag].mean() * 100
            
            row[f'{diag}_in_Missing(%)'] = round(prob_missing, 2)
            row[f'{diag}_in_Present(%)'] = round(prob_present, 2)
            
            # Simple ratio to see enrichment
            if prob_present > 0:
                enrichment = prob_missing / prob_present
            else:
                enrichment = 0
            row[f'{diag}_Ratio'] = round(enrichment, 2)
            
        results.append(row)
        
    results_df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ['Feature', 'Missing_Count']
    for diag in diag_cols:
        cols.append(f'{diag}_in_Missing(%)')
        # cols.append(f'{diag}_in_Present(%)') # Optional: hide present to reduce clutter
        cols.append(f'{diag}_Ratio')
        
    final_table = results_df[cols]
    
    print(final_table.to_string(index=False))
    final_table.to_csv(os.path.join(OUTPUT_DIR, 'missingness_diagnosis_correlation.csv'), index=False)
    print(f"\nSaved correlation table to {os.path.join(OUTPUT_DIR, 'missingness_diagnosis_correlation.csv')}")
    
    # 4. Heatmap of Missingness Co-occurrence
    # Check if missing features tend to be missing together
    missing_flags = [f'missing_{c}' for c in missing_cols]
    corr_matrix = df[missing_flags].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation between Missing Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'missingness_cooccurrence_heatmap.png'))
    print(f"Saved co-occurrence heatmap to {os.path.join(OUTPUT_DIR, 'missingness_cooccurrence_heatmap.png')}")

if __name__ == "__main__":
    main()
