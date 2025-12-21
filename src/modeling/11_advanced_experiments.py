import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
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

def get_per_class_metrics(y_true, y_pred, target_cols):
    metrics = {}
    for i, col in enumerate(target_cols):
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[col] = {'Precision': prec, 'Recall': rec, 'F1': f1}
    return metrics

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
    per_class_all = []
    
    # ================= 1. Baseline (for comparison) =================
    print("\n--- 1. Baseline (RF + Top-200) ---")
    clf_base = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_base.fit(X_train, y_train)
    
    y_val_prob = np.array([p[:, 1] for p in clf_base.predict_proba(X_val)]).T
    y_test_prob = np.array([p[:, 1] for p in clf_base.predict_proba(X_test)]).T
    
    thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
    y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
    
    acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred, y_test_prob)
    results_all.append({'Model': 'Baseline', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1, 'ROC_AUC': roc})
    
    pc = get_per_class_metrics(y_test.values, y_pred, target_cols)
    for col in target_cols:
        per_class_all.append({'Model': 'Baseline', 'Class': col, **pc[col]})
    
    print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    
    # ================= 2. Calibrated RF =================
    print("\n--- 2. Calibrated RF (Isotonic) ---")
    clf_cal = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_cal.fit(X_train, y_train)
    
    # Calibrate using validation set - need to calibrate each output separately for multi-label
    y_val_prob_cal = np.zeros((len(X_val), len(target_cols)))
    y_test_prob_cal = np.zeros((len(X_test), len(target_cols)))
    
    for i, col in enumerate(target_cols):
        # Wrap with calibration
        cal = CalibratedClassifierCV(clf_cal, method='isotonic', cv='prefit')
        # We need a single-output for calibration, so we create a wrapper
        # Simpler approach: calibrate based on RF probabilities directly
        # Use Platt scaling on the raw probabilities
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        prob_train = np.array([p[:, 1] for p in clf_cal.predict_proba(X_train)]).T[:, i].reshape(-1, 1)
        lr.fit(prob_train, y_train[col])
        
        prob_val = np.array([p[:, 1] for p in clf_cal.predict_proba(X_val)]).T[:, i].reshape(-1, 1)
        prob_test = np.array([p[:, 1] for p in clf_cal.predict_proba(X_test)]).T[:, i].reshape(-1, 1)
        
        y_val_prob_cal[:, i] = lr.predict_proba(prob_val)[:, 1]
        y_test_prob_cal[:, i] = lr.predict_proba(prob_test)[:, 1]
    
    thresholds_cal = find_optimal_thresholds(y_val, y_val_prob_cal, target_cols)
    y_pred_cal = apply_thresholds(y_test_prob_cal, thresholds_cal, target_cols)
    
    acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred_cal, y_test_prob_cal)
    results_all.append({'Model': 'Calibrated', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1, 'ROC_AUC': roc})
    
    pc = get_per_class_metrics(y_test.values, y_pred_cal, target_cols)
    for col in target_cols:
        per_class_all.append({'Model': 'Calibrated', 'Class': col, **pc[col]})
    
    print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    
    # ================= 3. 10-Seed Ensemble =================
    print("\n--- 3. 10-Seed Ensemble ---")
    n_seeds = 10
    y_prob_sum = np.zeros((len(X_test), len(target_cols)))
    y_val_prob_sum = np.zeros((len(X_val), len(target_cols)))
    
    class_counts = y_train.sum().sort_values()
    min_count = int(class_counts.iloc[0])
    sorted_classes = class_counts.index.tolist()
    
    def get_sampling_label(row):
        for cls in sorted_classes:
            if row[cls] == 1:
                return cls
        return 'NORM'
    
    sampling_labels = y_train.apply(get_sampling_label, axis=1)
    
    for seed in range(n_seeds):
        indices_to_keep = []
        for cls in sorted_classes:
            cls_indices = sampling_labels[sampling_labels == cls].index
            np.random.seed(seed)
            if len(cls_indices) >= min_count:
                selected = np.random.choice(cls_indices, min_count, replace=False)
            else:
                selected = cls_indices
            indices_to_keep.extend(selected)
        
        X_resampled = X_train.loc[indices_to_keep]
        y_resampled = y_train.loc[indices_to_keep]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        clf.fit(X_resampled, y_resampled)
        
        probs_val = np.array([p[:, 1] for p in clf.predict_proba(X_val)]).T
        probs_test = np.array([p[:, 1] for p in clf.predict_proba(X_test)]).T
        
        y_val_prob_sum += probs_val
        y_prob_sum += probs_test
    
    y_val_prob_ens = y_val_prob_sum / n_seeds
    y_test_prob_ens = y_prob_sum / n_seeds
    
    thresholds_ens = find_optimal_thresholds(y_val, y_val_prob_ens, target_cols)
    y_pred_ens = apply_thresholds(y_test_prob_ens, thresholds_ens, target_cols)
    
    acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred_ens, y_test_prob_ens)
    results_all.append({'Model': '10-Seed Ensemble', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1, 'ROC_AUC': roc})
    
    pc = get_per_class_metrics(y_test.values, y_pred_ens, target_cols)
    for col in target_cols:
        per_class_all.append({'Model': '10-Seed Ensemble', 'Class': col, **pc[col]})
    
    print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    
    # ================= Save Results =================
    results_df = pd.DataFrame(results_all)
    print("\n" + "="*90)
    print("ADVANCED MODEL COMPARISON")
    print("="*90)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'advanced_model_comparison.csv'), index=False)
    
    per_class_df = pd.DataFrame(per_class_all)
    per_class_df.to_csv(os.path.join(RESULTS_DIR, 'advanced_per_class_metrics.csv'), index=False)
    print(f"\nSaved results to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
