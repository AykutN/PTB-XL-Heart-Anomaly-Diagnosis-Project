import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import os

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
INPUT_PATH = os.path.join(BASE_DIR, 'data/processed/train_imputed.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports/feature_selection')

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH, index_col='ecg_id')
    
    # Define targets and features
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Drop targets and non-feature columns if any (like strat_fold, etc.)
    # We kept strat_fold in the previous step? Let's check columns.
    # We should exclude 'strat_fold' and targets from X
    drop_cols = target_cols + ['strat_fold', 'scp_codes', 'diagnostic_superclass']
    # Also drop any other metadata if present (patient_id etc were dropped earlier but check)
    # The cleaned file had some metadata dropped.
    
    # Safe drop of known non-feature columns
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Select only numeric columns to avoid string errors (e.g., device, report, etc.)
    X = X.select_dtypes(include=[np.number])
    
    y = df[target_cols]
    
    print(f"Initial Numeric Feature Count: {X.shape[1]}")
    
    # =================================================================================================
    # 1. VarianceThreshold
    # =================================================================================================
    print("\n--- Step 1: VarianceThreshold ---")
    
    # Calculate variances manually for plotting before filtering
    variances = X.var()
    
    # Filter for plotting (Zoom in to 0-1 range to avoid extreme outliers)
    variances_plot = variances[variances < 1]
    n_outliers = len(variances) - len(variances_plot)
    print(f"Plotting variance histogram for {len(variances_plot)} features (excluded {n_outliers} outliers > 1)")

    # Plot Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(variances_plot, kde=True, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=0.05, color='red', linestyle='--', label='Threshold (0.05)')
    plt.title('Öznitelik Varyans Dağılımı (0-1 Aralığı)')
    plt.xlabel('Varyans')
    plt.ylabel('Öznitelik Sayısı')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'variance_histogram.png'), dpi=300)
    print(f"Saved variance histogram to {os.path.join(OUTPUT_DIR, 'variance_histogram.png')}")
    
    # Apply Threshold
    selector = VarianceThreshold(threshold=0.05)
    X_var = selector.fit_transform(X)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    X_selected_var = pd.DataFrame(X_var, columns=selected_features, index=X.index)
    
    n_removed = X.shape[1] - X_selected_var.shape[1]
    print(f"Features Removed: {n_removed}")
    print(f"Features Remaining: {X_selected_var.shape[1]}")
    
    # =================================================================================================
    # 2. Correlation-based Filtering
    # =================================================================================================
    print("\n--- Step 2: Correlation-based Filtering ---")
    
    # Calculate Correlation Matrix
    corr_matrix = X_selected_var.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    print(f"High Correlation Pairs (>0.95) found: {len(to_drop)} features to drop.")
    
    # Drop features
    X_final = X_selected_var.drop(columns=to_drop)
    
    print(f"Features Removed by Correlation: {len(to_drop)}")
    print(f"Final Feature Count: {X_final.shape[1]}")
    
    # Plot Heatmap for a subset (Top 100 of the remaining features for visualization)
    # Since we don't have importance yet, we just take the first 100
    subset_cols = X_final.columns[:100]
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_final[subset_cols].corr(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('İlk 100 Öznitelik Korelasyon Matrisi (Filtreleme Sonrası)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap_sample.png'), dpi=300)
    print(f"Saved correlation heatmap to {os.path.join(OUTPUT_DIR, 'correlation_heatmap_sample.png')}")
    
    # =================================================================================================
    # 3. Random Forest Importance
    # =================================================================================================
    print("\n--- Step 3: Random Forest Importance ---")
    
    rf = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_final, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create DataFrame for importances
    feature_importance_df = pd.DataFrame({
        'Feature': X_final.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save full importance list
    feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importances.csv'), index=False)
    
    # Plot Top 30
    top_30 = feature_importance_df.head(30)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_30, palette='viridis')
    plt.title('En Önemli 30 Öznitelik (Random Forest)')
    plt.xlabel('Önem Skoru (Importance)')
    plt.ylabel('Öznitelik')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_importance_top30.png'), dpi=300)
    print(f"Saved RF importance plot to {os.path.join(OUTPUT_DIR, 'rf_importance_top30.png')}")
    
    print("\nTop 30 Features:")
    print(top_30.to_string(index=False))
    
    # =================================================================================================
    # 4. Top-N Feature Sets
    # =================================================================================================
    print("\n--- Step 4: Exporting Top-N Sets ---")
    
    for n in [50, 100, 200]:
        top_n_df = feature_importance_df.head(n)
        output_file = os.path.join(OUTPUT_DIR, f'top{n}_features.csv')
        top_n_df.to_csv(output_file, index=False)
        print(f"Saved Top-{n} features to {output_file}")

if __name__ == "__main__":
    main()
