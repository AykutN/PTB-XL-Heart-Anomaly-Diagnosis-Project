import pandas as pd
import numpy as np
import os
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FEATURE_DIR = os.path.join(BASE_DIR, 'reports/feature_selection/')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/')

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_imputed.csv'), index_col='ecg_id')
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_imputed.csv'), index_col='ecg_id')
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_imputed.csv'), index_col='ecg_id')
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    return train_df, val_df, test_df, target_cols

def load_features(n):
    path = os.path.join(FEATURE_DIR, f'top{n}_features.csv')
    df = pd.read_csv(path)
    return df['Feature'].tolist()

def find_optimal_thresholds(y_val, y_val_prob, target_cols):
    thresholds = {}
    for i, col in enumerate(target_cols):
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred = (y_val_prob[:, i] >= threshold).astype(int)
            f1 = f1_score(y_val[col], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[col] = best_threshold
    return thresholds

def apply_thresholds(y_prob, thresholds, target_cols):
    y_pred = np.zeros_like(y_prob)
    for i, col in enumerate(target_cols):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[col]).astype(int)
    return y_pred.astype(int)

def evaluate_model(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob, average='macro')
    except:
        roc = 0
    return acc, prec, rec, f1, roc

def main():
    print("Loading data...")
    train_df, val_df, test_df, target_cols = load_data()
    features = load_features(200)
    
    X_train = train_df[features]
    y_train = train_df[target_cols]
    X_val = val_df[features]
    y_val = val_df[target_cols]
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    results_all = []
    
    # ================= 1. Standard RF (Baseline) =================
    print("\n--- 1. Standard RF (Baseline) ---")
    clf_base = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_base.fit(X_train, y_train)
    
    y_val_prob = np.array([p[:, 1] for p in clf_base.predict_proba(X_val)]).T
    y_test_prob = np.array([p[:, 1] for p in clf_base.predict_proba(X_test)]).T
    
    thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
    y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
    
    acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred, y_test_prob)
    results_all.append({'Model': 'Standard RF', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1, 'ROC_AUC': roc})
    print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    
    # ================= 2. BalancedRandomForest =================
    print("\n--- 2. BalancedRandomForest ---")
    # Note: BalancedRF doesn't support multi-output directly
    # We need to train one per class (One-vs-Rest approach)
    y_val_prob_brf = np.zeros((len(X_val), len(target_cols)))
    y_test_prob_brf = np.zeros((len(X_test), len(target_cols)))
    
    for i, col in enumerate(target_cols):
        print(f"  Training for {col}...")
        clf_brf = BalancedRandomForestClassifier(
            n_estimators=250, 
            max_depth=12, 
            random_state=42, 
            n_jobs=-1,
            sampling_strategy='all'
        )
        clf_brf.fit(X_train, y_train[col])
        
        y_val_prob_brf[:, i] = clf_brf.predict_proba(X_val)[:, 1]
        y_test_prob_brf[:, i] = clf_brf.predict_proba(X_test)[:, 1]
    
    thresholds_brf = find_optimal_thresholds(y_val, y_val_prob_brf, target_cols)
    y_pred_brf = apply_thresholds(y_test_prob_brf, thresholds_brf, target_cols)
    
    acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred_brf, y_test_prob_brf)
    results_all.append({'Model': 'BalancedRandomForest', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1, 'ROC_AUC': roc})
    print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    
    # ================= 3. RF with n_estimators=500 =================
    print("\n--- 3. RF with n_estimators=500 ---")
    clf_500 = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_500.fit(X_train, y_train)
    
    y_val_prob_500 = np.array([p[:, 1] for p in clf_500.predict_proba(X_val)]).T
    y_test_prob_500 = np.array([p[:, 1] for p in clf_500.predict_proba(X_test)]).T
    
    thresholds_500 = find_optimal_thresholds(y_val, y_val_prob_500, target_cols)
    y_pred_500 = apply_thresholds(y_test_prob_500, thresholds_500, target_cols)
    
    acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred_500, y_test_prob_500)
    results_all.append({'Model': 'RF (n=500)', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1, 'ROC_AUC': roc})
    print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    
    # ================= Save Results =================
    results_df = pd.DataFrame(results_all)
    print("\n" + "="*90)
    print("BALANCED RF COMPARISON")
    print("="*90)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'balanced_rf_comparison.csv'), index=False)

if __name__ == "__main__":
    main()
