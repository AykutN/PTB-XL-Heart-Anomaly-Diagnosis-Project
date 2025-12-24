import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data/processed/train_imputed.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports/feature_selection')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH, index_col='ecg_id')
    
    # Define targets
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Drop targets and non-feature columns
    drop_cols = target_cols + ['strat_fold', 'scp_codes', 'diagnostic_superclass']
    
    # Prepare X and y
    # Select only numeric columns to avoid string errors
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    y = df[target_cols]
    
    print(f"Initial Numeric Feature Count: {X.shape[1]}")
    
    # =================================================================================================
    # 1. VarianceThreshold
    # =================================================================================================
    print("\n--- Step 1: VarianceThreshold (0.05) ---")
    
    selector = VarianceThreshold(threshold=0.05)
    X_var = selector.fit_transform(X)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    X_selected = pd.DataFrame(X_var, columns=selected_features, index=X.index)
    
    n_removed = X.shape[1] - X_selected.shape[1]
    print(f"Features Removed: {n_removed}")
    print(f"Features Remaining: {X_selected.shape[1]}")
    
    # =================================================================================================
    # 2. Random Forest Importance
    # =================================================================================================
    print("\n--- Step 2: Random Forest Importance ---")
    
    # Train RF on the filtered feature set
    rf = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_selected, y)
    
    importances = rf.feature_importances_
    
    # Create DataFrame for importances
    feature_importance_df = pd.DataFrame({
        'Feature': X_selected.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save full importance list
    feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, 'rf_feature_importances.csv'), index=False)
    
    # Plot Top 30
    top_30 = feature_importance_df.head(30)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_30, palette='viridis')
    plt.title('En Önemli 30 Öznitelik (Random Forest)')
    plt.xlabel('Önem Skoru (Importance)')
    plt.ylabel('Öznitelik')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_importance_top30_final.png'), dpi=300)
    print(f"Saved RF importance plot to {os.path.join(OUTPUT_DIR, 'rf_importance_top30_final.png')}")
    
    # =================================================================================================
    # 3. Export Top-N Sets
    # =================================================================================================
    print("\n--- Step 3: Exporting Top-N Sets ---")
    
    for n in [50, 100, 200]:
        top_n_df = feature_importance_df.head(n)
        output_file = os.path.join(OUTPUT_DIR, f'top{n}_features.csv')
        top_n_df.to_csv(output_file, index=False)
        print(f"Saved Top-{n} features to {output_file}")
        
    print("Feature selection completed.")

if __name__ == "__main__":
    main()
