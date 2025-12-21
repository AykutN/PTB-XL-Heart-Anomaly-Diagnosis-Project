import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FEATURE_DIR = os.path.join(BASE_DIR, 'reports/feature_selection/')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/')
REPORT_DIR = os.path.join(BASE_DIR, 'reports/model_comparison/')

os.makedirs(REPORT_DIR, exist_ok=True)

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
            f1 = f1_score(y_val[col], y_pred)
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
    
    results_all = []
    roc_data = {}
    
    # ================= Standard Models with Threshold Optimization =================
    feature_sets = [('Top-50', 50), ('Top-100', 100), ('Top-200', 200)]
    
    for set_name, n_features in feature_sets:
        print(f"\n--- Training RF + {set_name} ---")
        features = load_features(n_features)
        
        X_train = train_df[features]
        y_train = train_df[target_cols]
        X_val = val_df[features]
        y_val = val_df[target_cols]
        X_test = test_df[features]
        y_test = test_df[target_cols]
        
        clf = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Get probabilities
        y_val_prob = np.array([p[:, 1] for p in clf.predict_proba(X_val)]).T
        y_test_prob = np.array([p[:, 1] for p in clf.predict_proba(X_test)]).T
        
        # Find optimal thresholds on validation set
        thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
        
        # Apply to test set
        y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
        
        acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred, y_test_prob)
        
        results_all.append({
            'Model': f'RF + {set_name}',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'ROC_AUC': roc
        })
        
        if n_features == 200:
            roc_data['RF + Top-200'] = (y_test.values, y_test_prob)
    
    # ================= Ensemble Undersampling Model =================
    print("\n--- Training RF + Top-200 + Ensemble ---")
    features = load_features(200)
    
    X_train = train_df[features]
    y_train = train_df[target_cols]
    X_val = val_df[features]
    y_val = val_df[target_cols]
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    # Ensemble parameters
    n_estimators_ensemble = 50
    y_prob_sum = np.zeros((len(X_test), len(target_cols)))
    
    # Find minority class count
    class_counts = y_train.sum().sort_values()
    min_count = int(class_counts.iloc[0])
    sorted_classes = class_counts.index.tolist()
    
    def get_sampling_label(row):
        for cls in sorted_classes:
            if row[cls] == 1:
                return cls
        return 'NORM'
    
    sampling_labels = y_train.apply(get_sampling_label, axis=1)
    
    for i in range(n_estimators_ensemble):
        indices_to_keep = []
        for cls in sorted_classes:
            cls_indices = sampling_labels[sampling_labels == cls].index
            if len(cls_indices) >= min_count:
                selected = np.random.choice(cls_indices, min_count, replace=False)
            else:
                selected = cls_indices
            indices_to_keep.extend(selected)
        
        X_resampled = X_train.loc[indices_to_keep]
        y_resampled = y_train.loc[indices_to_keep]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=i, n_jobs=-1)
        clf.fit(X_resampled, y_resampled)
        
        probs = np.array([p[:, 1] for p in clf.predict_proba(X_test)]).T
        y_prob_sum += probs
    
    y_test_prob_ens = y_prob_sum / n_estimators_ensemble
    
    # Get validation probs for threshold optimization (retrain one model for this)
    clf_val = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf_val.fit(X_train, y_train)
    y_val_prob_ens = np.array([p[:, 1] for p in clf_val.predict_proba(X_val)]).T
    
    thresholds_ens = find_optimal_thresholds(y_val, y_val_prob_ens, target_cols)
    y_pred_ens = apply_thresholds(y_test_prob_ens, thresholds_ens, target_cols)
    
    acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred_ens, y_test_prob_ens)
    
    results_all.append({
        'Model': 'RF + Top-200 + Ensemble',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'ROC_AUC': roc
    })
    
    roc_data['RF + Top-200 + Ensemble'] = (y_test.values, y_test_prob_ens)
    
    # ================= Save Results =================
    results_df = pd.DataFrame(results_all)
    print("\n" + "="*90)
    print("FULL MODEL COMPARISON TABLE (All with Threshold Optimization)")
    print("="*90)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'full_model_comparison.csv'), index=False)
    
    # ================= Plot ROC Curves =================
    print("\nPlotting ROC curves...")
    plt.figure(figsize=(10, 8))
    
    colors = {'NORM': '#1f77b4', 'MI': '#ff7f0e', 'STTC': '#2ca02c', 'CD': '#d62728', 'HYP': '#9467bd'}
    
    # Use the best model's probabilities (Ensemble)
    y_true, y_prob = roc_data['RF + Top-200 + Ensemble']
    
    for i, col in enumerate(target_cols):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[col], lw=2, label=f'{col} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
    plt.ylabel('Doğru Pozitif Oranı (TPR)', fontsize=12)
    plt.title('ROC Eğrileri (En İyi Model: RF + Top-200 + Ensemble)', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'roc_curves_final.png'), dpi=300)
    print(f"Saved ROC curves to {os.path.join(REPORT_DIR, 'roc_curves_final.png')}")

if __name__ == "__main__":
    main()
