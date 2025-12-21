import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FEATURE_DIR = os.path.join(BASE_DIR, 'reports/feature_selection/')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/')

def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_imputed.csv'), index_col='ecg_id')
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_imputed.csv'), index_col='ecg_id')
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_imputed.csv'), index_col='ecg_id')
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    return train_df, val_df, test_df, target_cols

def load_features(n):
    path = os.path.join(FEATURE_DIR, f'top{n}_features.csv')
    df = pd.read_csv(path)
    return df['Feature'].tolist()

def find_optimal_threshold(y_true, y_prob, target_name):
    """Find the threshold that maximizes F1 score for a single class."""
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    return best_threshold, best_f1

def main():
    train_df, val_df, test_df, target_cols = load_data()
    features = load_features(200)
    
    X_train = train_df[features]
    y_train = train_df[target_cols]
    
    X_val = val_df[features]
    y_val = val_df[target_cols]
    
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    # Train the model
    print("Training RF with class_weight='balanced'...")
    clf = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Get probabilities on Validation set
    print("Finding optimal thresholds on Validation set...")
    y_val_prob = np.array(clf.predict_proba(X_val))
    y_val_prob_formatted = np.array([p[:, 1] for p in y_val_prob]).T
    
    optimal_thresholds = {}
    for i, col in enumerate(target_cols):
        threshold, f1 = find_optimal_threshold(y_val[col], y_val_prob_formatted[:, i], col)
        optimal_thresholds[col] = threshold
        print(f"  {col}: Optimal Threshold = {threshold:.2f}, Val F1 = {f1:.4f}")
        
    # Apply optimal thresholds on Test set
    print("\nApplying optimized thresholds on Test set...")
    y_test_prob = np.array(clf.predict_proba(X_test))
    y_test_prob_formatted = np.array([p[:, 1] for p in y_test_prob]).T
    
    # Default predictions (threshold=0.5)
    y_pred_default = (y_test_prob_formatted >= 0.5).astype(int)
    
    # Optimized predictions
    y_pred_optimized = np.zeros_like(y_pred_default)
    for i, col in enumerate(target_cols):
        y_pred_optimized[:, i] = (y_test_prob_formatted[:, i] >= optimal_thresholds[col]).astype(int)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("COMPARISON: Default (0.5) vs Optimized Thresholds")
    print("="*60)
    
    # Convert to DataFrame for easier metric calculation
    y_test_np = y_test.values
    
    # Default Metrics
    acc_default = accuracy_score(y_test_np, y_pred_default)
    f1_macro_default = f1_score(y_test_np, y_pred_default, average='macro')
    roc_auc_default = roc_auc_score(y_test_np, y_test_prob_formatted, average='macro')
    
    # Optimized Metrics
    acc_optimized = accuracy_score(y_test_np, y_pred_optimized)
    f1_macro_optimized = f1_score(y_test_np, y_pred_optimized, average='macro')
    roc_auc_optimized = roc_auc_score(y_test_np, y_test_prob_formatted, average='macro') # Same probs, same AUC
    
    print(f"{'Metric':<20} | {'Default (0.5)':<15} | {'Optimized':<15} | {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<20} | {acc_default:<15.4f} | {acc_optimized:<15.4f} | {acc_optimized - acc_default:+.4f}")
    print(f"{'Macro F1':<20} | {f1_macro_default:<15.4f} | {f1_macro_optimized:<15.4f} | {f1_macro_optimized - f1_macro_default:+.4f}")
    print(f"{'ROC-AUC':<20} | {roc_auc_default:<15.4f} | {roc_auc_optimized:<15.4f} | {roc_auc_optimized - roc_auc_default:+.4f}")
    
    # Per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS METRICS (Optimized Thresholds)")
    print("="*60)
    
    results = []
    for i, col in enumerate(target_cols):
        prec = precision_score(y_test_np[:, i], y_pred_optimized[:, i])
        rec = recall_score(y_test_np[:, i], y_pred_optimized[:, i])
        f1 = f1_score(y_test_np[:, i], y_pred_optimized[:, i])
        results.append({
            'Class': col,
            'Threshold': optimal_thresholds[col],
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })
        print(f"{col:<10} | Threshold: {optimal_thresholds[col]:.2f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'optimized_thresholds_results.csv'), index=False)
    print(f"\nSaved optimized results to {os.path.join(RESULTS_DIR, 'optimized_thresholds_results.csv')}")
    
    # Save comparison summary
    summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Macro_F1', 'ROC_AUC'],
        'Default_0.5': [acc_default, f1_macro_default, roc_auc_default],
        'Optimized': [acc_optimized, f1_macro_optimized, roc_auc_optimized],
        'Improvement': [acc_optimized - acc_default, f1_macro_optimized - f1_macro_default, 0]
    })
    summary.to_csv(os.path.join(RESULTS_DIR, 'threshold_optimization_summary.csv'), index=False)

if __name__ == "__main__":
    main()
