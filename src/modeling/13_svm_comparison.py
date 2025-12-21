import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
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
    
    # Identify numeric columns for "All Features"
    drop_cols = target_cols + ['strat_fold', 'scp_codes', 'diagnostic_superclass']
    all_features = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).select_dtypes(include=[np.number]).columns.tolist()
    
    top200_features = load_features(200)
    
    results_all = []
    
    experiments = [
        ("SVM (All Features)", all_features),
        ("SVM (Top-200)", top200_features)
    ]
    
    for exp_name, features in experiments:
        print(f"\n--- Running {exp_name} ---")
        
        X_train = train_df[features]
        y_train = train_df[target_cols]
        X_val = val_df[features]
        y_val = val_df[target_cols]
        X_test = test_df[features]
        y_test = test_df[target_cols]
        
        # Scaling is crucial for SVM
        print("  Scaling data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM (OneVsRest is needed for multi-label)
        print("  Training SVM...")
        # Using probability=True is slow but needed for threshold optimization
        # Using a smaller cache_size might be needed if memory is tight, but default is usually fine.
        # class_weight='balanced' is important
        clf = OneVsRestClassifier(SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42, cache_size=1000))
        clf.fit(X_train_scaled, y_train)
        
        print("  Predicting...")
        y_val_prob = clf.predict_proba(X_val_scaled)
        y_test_prob = clf.predict_proba(X_test_scaled)
        
        print("  Optimizing thresholds...")
        thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
        y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
        
        acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred, y_test_prob)
        results_all.append({'Model': exp_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1, 'ROC_AUC': roc})
        print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, ROC-AUC: {roc:.4f}")

    # ================= Save Results =================
    results_df = pd.DataFrame(results_all)
    print("\n" + "="*90)
    print("SVM COMPARISON RESULTS")
    print("="*90)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'svm_comparison.csv'), index=False)

if __name__ == "__main__":
    main()
