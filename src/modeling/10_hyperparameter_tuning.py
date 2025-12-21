import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

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
    features = load_features(200)
    
    X_train = train_df[features]
    y_train = train_df[target_cols]
    X_val = val_df[features]
    y_val = val_df[target_cols]
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    # ================= 1. Visualize Class Weights =================
    print("\n--- Generating Class Weights Visualization ---")
    
    # Calculate balanced class weights for each target
    # Formula: n_samples / (n_classes * n_positive)
    n_samples = len(y_train)
    class_counts = y_train.sum()
    
    # Balanced weights (sklearn formula)
    balanced_weights = {}
    for col in target_cols:
        n_positive = class_counts[col]
        n_negative = n_samples - n_positive
        # Weight for positive class
        weight = n_samples / (2 * n_positive)
        balanced_weights[col] = weight
    
    # Custom weights (Extra emphasis on HYP and CD)
    custom_weights = balanced_weights.copy()
    custom_weights['HYP'] *= 1.5  # 50% extra
    custom_weights['CD'] *= 1.3   # 30% extra
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(target_cols))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [balanced_weights[c] for c in target_cols], width, label='Balanced', color='steelblue')
    bars2 = ax.bar(x + width/2, [custom_weights[c] for c in target_cols], width, label='Custom (HYP+50%, CD+30%)', color='darkorange')
    
    ax.set_ylabel('Sınıf Ağırlığı')
    ax.set_xlabel('Tanı Sınıfı')
    ax.set_title('Sınıf Dengesizliği Ağırlıkları')
    ax.set_xticks(x)
    ax.set_xticklabels(target_cols)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'class_weights_comparison.png'), dpi=300)
    print(f"Saved class weights plot to {os.path.join(REPORT_DIR, 'class_weights_comparison.png')}")
    
    # ================= 2. Hyperparameter Tuning =================
    print("\n--- Running Hyperparameter Experiments ---")
    
    results = []
    
    # Baseline
    configs = [
        {"name": "Baseline", "params": {"n_estimators": 250, "max_depth": 12, "class_weight": "balanced"}},
        {"name": "min_samples_leaf=5", "params": {"n_estimators": 250, "max_depth": 12, "class_weight": "balanced", "min_samples_leaf": 5}},
        {"name": "min_samples_leaf=10", "params": {"n_estimators": 250, "max_depth": 12, "class_weight": "balanced", "min_samples_leaf": 10}},
        {"name": "max_features=sqrt", "params": {"n_estimators": 250, "max_depth": 12, "class_weight": "balanced", "max_features": "sqrt"}},
        {"name": "max_features=log2", "params": {"n_estimators": 250, "max_depth": 12, "class_weight": "balanced", "max_features": "log2"}},
    ]
    
    # Add custom weight config
    # For multi-output, we need to pass a dict per output
    # But sklearn RF doesn't directly support this for multi-output.
    # We'll use sample_weight instead for custom weighting later if needed.
    # For now, let's stick with the structural hyperparameters.
    
    for config in configs:
        print(f"  Testing: {config['name']}")
        
        clf = RandomForestClassifier(**config['params'], random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        y_val_prob = np.array([p[:, 1] for p in clf.predict_proba(X_val)]).T
        y_test_prob = np.array([p[:, 1] for p in clf.predict_proba(X_test)]).T
        
        thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
        y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
        
        acc, prec, rec, f1, roc = evaluate_model(y_test.values, y_pred, y_test_prob)
        
        results.append({
            'Model': config['name'],
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'ROC_AUC': roc
        })
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*90)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*90)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'hyperparameter_tuning_results.csv'), index=False)

if __name__ == "__main__":
    main()
