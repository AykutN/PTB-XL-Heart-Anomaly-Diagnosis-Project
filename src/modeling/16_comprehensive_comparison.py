"""
Comprehensive Model Comparison: RF and SVM with Class-Weight and Ensemble Undersampling
Includes Top-50, Top-100, Top-200 feature sets and ensemble undersampling approach.
Generates visualizations for paper.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FEATURE_DIR = os.path.join(BASE_DIR, 'reports/feature_selection/')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/')
REPORT_DIR = os.path.join(BASE_DIR, 'reports/model_comparison/')

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Load train, validation, and test datasets."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_imputed.csv'), index_col='ecg_id')
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_imputed.csv'), index_col='ecg_id')
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_imputed.csv'), index_col='ecg_id')
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    return train_df, val_df, test_df, target_cols

def load_features(n):
    """Load top N features from feature selection results."""
    path = os.path.join(FEATURE_DIR, f'top{n}_features.csv')
    df = pd.read_csv(path)
    return df['Feature'].tolist()

def find_optimal_thresholds(y_val, y_val_prob, target_cols):
    """Find optimal classification thresholds for each class."""
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
    """Apply optimal thresholds to probability predictions."""
    y_pred = np.zeros_like(y_prob)
    for i, col in enumerate(target_cols):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[col]).astype(int)
    return y_pred.astype(int)

def evaluate_model_macro(y_true, y_pred, y_prob):
    """Calculate macro-averaged metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob, average='macro')
    except:
        roc = 0
    return acc, prec, rec, f1, roc

def train_rf_class_weight(X_train, y_train, X_val, y_val, X_test, y_test, target_cols):
    """Train Random Forest with class weights."""
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Get probabilities
    y_val_prob = np.array([p[:, 1] for p in clf.predict_proba(X_val)]).T
    y_test_prob = np.array([p[:, 1] for p in clf.predict_proba(X_test)]).T
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
    y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
    
    return y_pred, y_test_prob

def train_svm_class_weight(X_train, y_train, X_val, y_val, X_test, y_test, target_cols):
    """Train SVM with class weights."""
    # Scale data for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Use OneVsRestClassifier for multi-label classification
    base_svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42, cache_size=1000)
    clf = OneVsRestClassifier(base_svm, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # Get probabilities
    y_val_prob = clf.predict_proba(X_val_scaled)
    y_test_prob = clf.predict_proba(X_test_scaled)
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
    y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
    
    return y_pred, y_test_prob

def train_rf_ensemble_undersampling(X_train, y_train, X_val, y_val, X_test, y_test, target_cols, n_estimators=50):
    """Train Random Forest with ensemble undersampling."""
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
    
    for i in range(n_estimators):
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
    
    y_test_prob_ens = y_prob_sum / n_estimators
    
    # Get validation probs for threshold optimization
    clf_val = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf_val.fit(X_train, y_train)
    y_val_prob_ens = np.array([p[:, 1] for p in clf_val.predict_proba(X_val)]).T
    
    thresholds = find_optimal_thresholds(y_val, y_val_prob_ens, target_cols)
    y_pred = apply_thresholds(y_test_prob_ens, thresholds, target_cols)
    
    return y_pred, y_test_prob_ens

def train_svm_ensemble_undersampling(X_train, y_train, X_val, y_val, X_test, y_test, target_cols, n_estimators=50):
    """Train SVM with ensemble undersampling."""
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
    
    # Global scaler (fit on full training set)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    for i in range(n_estimators):
        if (i + 1) % 10 == 0:
            print(f"    Ensemble iteration {i+1}/{n_estimators}...")
        
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
        
        # Scale resampled data
        X_resampled_scaled = scaler.transform(X_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        base_svm = SVC(kernel='rbf', probability=True, class_weight=None, random_state=i, cache_size=500)
        clf = OneVsRestClassifier(base_svm, n_jobs=-1)
        clf.fit(X_resampled_scaled, y_resampled)
        
        probs = clf.predict_proba(X_test_scaled)
        y_prob_sum += probs
    
    y_test_prob_ens = y_prob_sum / n_estimators
    
    # Get validation probs for threshold optimization
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    base_svm_val = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42, cache_size=1000)
    clf_val = OneVsRestClassifier(base_svm_val, n_jobs=-1)
    clf_val.fit(X_train_scaled, y_train)
    y_val_prob_ens = clf_val.predict_proba(X_val_scaled)
    
    thresholds = find_optimal_thresholds(y_val, y_val_prob_ens, target_cols)
    y_pred = apply_thresholds(y_test_prob_ens, thresholds, target_cols)
    
    return y_pred, y_test_prob_ens

def create_class_weight_comparison_visualization(results_df):
    """Create visualization for class-weight approach across feature sets."""
    print("\nCreating class-weight comparison visualization...")
    
    # Filter class-weight results
    cw_results = results_df[results_df['Approach'] == 'Class-Weight'].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Separate by model
        rf_data = cw_results[cw_results['Model'] == 'Random Forest']
        svm_data = cw_results[cw_results['Model'] == 'SVM']
        
        x = np.arange(len(['Top-50', 'Top-100', 'Top-200']))
        width = 0.35
        
        rf_values = [rf_data[rf_data['Features'] == f'Top-{n}'][metric].values[0] 
                    if len(rf_data[rf_data['Features'] == f'Top-{n}']) > 0 else 0 
                    for n in [50, 100, 200]]
        svm_values = [svm_data[svm_data['Features'] == f'Top-{n}'][metric].values[0] 
                     if len(svm_data[svm_data['Features'] == f'Top-{n}']) > 0 else 0 
                     for n in [50, 100, 200]]
        
        bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', 
                      color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, svm_values, width, label='SVM', 
                      color='#e74c3c', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} - Class-Weight Approach', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Top-50', 'Top-100', 'Top-200'])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.suptitle('Model Performance Comparison: Class-Weight Approach Across Feature Sets', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'class_weight_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved class-weight comparison to {os.path.join(REPORT_DIR, 'class_weight_comparison.png')}")
    plt.close()

def main():
    """Main function to run comprehensive comparison."""
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON: RF vs SVM")
    print("Class-Weight (Top-50, Top-100, Top-200) + Ensemble Undersampling (Top-200)")
    print("="*80)
    
    # Load data
    train_df, val_df, test_df, target_cols = load_data()
    
    print(f"\nLoaded data:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    results_all = []
    
    # ================= Class-Weight Approach =================
    print("\n" + "="*80)
    print("CLASS-WEIGHT APPROACH")
    print("="*80)
    
    feature_sets = [('Top-50', 50), ('Top-100', 100), ('Top-200', 200)]
    
    for set_name, n_features in feature_sets:
        print(f"\n--- {set_name} Features ---")
        features = load_features(n_features)
        
        X_train = train_df[features]
        y_train = train_df[target_cols]
        X_val = val_df[features]
        y_val = val_df[target_cols]
        X_test = test_df[features]
        y_test = test_df[target_cols]
        
        # Random Forest
        print(f"  Training RF + {set_name} + Class-Weight...")
        rf_pred, rf_prob = train_rf_class_weight(
            X_train, y_train, X_val, y_val, X_test, y_test, target_cols
        )
        rf_metrics = evaluate_model_macro(y_test.values, rf_pred, rf_prob)
        results_all.append({
            'Model': 'Random Forest',
            'Features': set_name,
            'Approach': 'Class-Weight',
            'Accuracy': rf_metrics[0],
            'Precision': rf_metrics[1],
            'Recall': rf_metrics[2],
            'F1_Score': rf_metrics[3],
            'ROC_AUC': rf_metrics[4]
        })
        print(f"    RF - F1: {rf_metrics[3]:.4f}, ROC-AUC: {rf_metrics[4]:.4f}")
        
        # SVM
        print(f"  Training SVM + {set_name} + Class-Weight...")
        svm_pred, svm_prob = train_svm_class_weight(
            X_train, y_train, X_val, y_val, X_test, y_test, target_cols
        )
        svm_metrics = evaluate_model_macro(y_test.values, svm_pred, svm_prob)
        results_all.append({
            'Model': 'SVM',
            'Features': set_name,
            'Approach': 'Class-Weight',
            'Accuracy': svm_metrics[0],
            'Precision': svm_metrics[1],
            'Recall': svm_metrics[2],
            'F1_Score': svm_metrics[3],
            'ROC_AUC': svm_metrics[4]
        })
        print(f"    SVM - F1: {svm_metrics[3]:.4f}, ROC-AUC: {svm_metrics[4]:.4f}")
    
    # ================= Ensemble Undersampling Approach =================
    print("\n" + "="*80)
    print("ENSEMBLE UNDERSAMPLING APPROACH (Top-200)")
    print("="*80)
    
    features = load_features(200)
    X_train = train_df[features]
    y_train = train_df[target_cols]
    X_val = val_df[features]
    y_val = val_df[target_cols]
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    # Random Forest Ensemble
    print("  Training RF + Top-200 + Ensemble Undersampling (50 models)...")
    rf_pred_ens, rf_prob_ens = train_rf_ensemble_undersampling(
        X_train, y_train, X_val, y_val, X_test, y_test, target_cols, n_estimators=50
    )
    rf_ens_metrics = evaluate_model_macro(y_test.values, rf_pred_ens, rf_prob_ens)
    results_all.append({
        'Model': 'Random Forest',
        'Features': 'Top-200',
        'Approach': 'Ensemble Undersampling',
        'Accuracy': rf_ens_metrics[0],
        'Precision': rf_ens_metrics[1],
        'Recall': rf_ens_metrics[2],
        'F1_Score': rf_ens_metrics[3],
        'ROC_AUC': rf_ens_metrics[4]
    })
    print(f"    RF Ensemble - F1: {rf_ens_metrics[3]:.4f}, ROC-AUC: {rf_ens_metrics[4]:.4f}")
    
    # SVM Ensemble
    print("  Training SVM + Top-200 + Ensemble Undersampling (50 models)...")
    print("  Note: This may take 1-2 hours due to SVM training time...")
    svm_pred_ens, svm_prob_ens = train_svm_ensemble_undersampling(
        X_train, y_train, X_val, y_val, X_test, y_test, target_cols, n_estimators=50
    )
    svm_ens_metrics = evaluate_model_macro(y_test.values, svm_pred_ens, svm_prob_ens)
    results_all.append({
        'Model': 'SVM',
        'Features': 'Top-200',
        'Approach': 'Ensemble Undersampling',
        'Accuracy': svm_ens_metrics[0],
        'Precision': svm_ens_metrics[1],
        'Recall': svm_ens_metrics[2],
        'F1_Score': svm_ens_metrics[3],
        'ROC_AUC': svm_ens_metrics[4]
    })
    print(f"    SVM Ensemble - F1: {svm_ens_metrics[3]:.4f}, ROC-AUC: {svm_ens_metrics[4]:.4f}")
    
    # ================= Save Results =================
    results_df = pd.DataFrame(results_all)
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'comprehensive_model_comparison.csv'), index=False)
    
    # ================= Create Visualizations =================
    create_class_weight_comparison_visualization(results_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Visualizations saved to: {REPORT_DIR}")

if __name__ == "__main__":
    main()

