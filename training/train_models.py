"""
Class-Weight Approach Comparison: RF vs SVM with Top-200 Features
Generates results and visualizations for paper.
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def evaluate_per_class(y_true, y_pred, y_prob, target_cols):
    """Calculate per-class metrics."""
    per_class_metrics = []
    for i, col in enumerate(target_cols):
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        try:
            roc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except:
            roc = 0
        
        per_class_metrics.append({
            'Class': col,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'ROC_AUC': roc
        })
    return pd.DataFrame(per_class_metrics)

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

def create_comparison_visualizations(results_df, per_class_rf, per_class_svm, target_cols):
    """Create comprehensive visualization plots."""
    print("\nCreating visualizations...")
    
    # 1. Overall metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        x = np.arange(len(results_df))
        width = 0.35
        
        rf_values = results_df[results_df['Model'] == 'Random Forest'][metric].values
        svm_values = results_df[results_df['Model'] == 'SVM'][metric].values
        
        if len(rf_values) > 0 and len(svm_values) > 0:
            bars1 = ax.bar(x[0] - width/2, rf_values[0], width, 
                          label='Random Forest', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x[0] + width/2, svm_values[0], width,
                          label='SVM', color='#e74c3c', alpha=0.8)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels(['Models'], rotation=0)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.suptitle('Model Performance Comparison: Random Forest vs SVM (Class-Weight, Top-200)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'rf_svm_class_weight_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved metrics comparison to {os.path.join(REPORT_DIR, 'rf_svm_class_weight_comparison.png')}")
    plt.close()
    
    # 2. Per-class F1 score comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(target_cols))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, per_class_rf['F1_Score'].values, width,
                  label='Random Forest', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, per_class_svm['F1_Score'].values, width,
                  label='SVM', color='#e74c3c', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Diagnostic Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Score Comparison (Class-Weight, Top-200)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(target_cols)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'per_class_f1_class_weight.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved per-class F1 comparison to {os.path.join(REPORT_DIR, 'per_class_f1_class_weight.png')}")
    plt.close()
    
    # 3. Per-class metrics heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (df, title, ax) in enumerate([(per_class_rf, 'Random Forest', axes[0]),
                                           (per_class_svm, 'SVM', axes[1])]):
        metrics_df = df.set_index('Class')[['Precision', 'Recall', 'F1_Score', 'ROC_AUC']]
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title(f'{title} - Per-Class Metrics (Class-Weight)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Diagnostic Class', fontsize=11)
    
    plt.suptitle('Per-Class Performance Metrics Heatmap (Class-Weight Approach)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'per_class_metrics_heatmap_class_weight.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved metrics heatmap to {os.path.join(REPORT_DIR, 'per_class_metrics_heatmap_class_weight.png')}")
    plt.close()

def main():
    """Main function to run class-weight comparison."""
    print("="*80)
    print("CLASS-WEIGHT APPROACH COMPARISON: Random Forest vs SVM")
    print("Top-200 Features")
    print("="*80)
    
    # Load data
    train_df, val_df, test_df, target_cols = load_data()
    features = load_features(200)  # Use Top-200 features
    
    print(f"\nLoaded data:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Features: {len(features)}")
    print(f"  Target classes: {', '.join(target_cols)}")
    
    # Prepare data
    X_train = train_df[features]
    y_train = train_df[target_cols]
    X_val = val_df[features]
    y_val = val_df[target_cols]
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    results_all = []
    per_class_results = []
    
    # ================= Train Random Forest =================
    print("\n" + "="*80)
    print("RANDOM FOREST MODEL (Class-Weight)")
    print("="*80)
    print("  Training Random Forest...")
    rf_pred, rf_prob = train_rf_class_weight(
        X_train, y_train, X_val, y_val, X_test, y_test, target_cols
    )
    
    rf_macro = evaluate_model_macro(y_test.values, rf_pred, rf_prob)
    rf_per_class = evaluate_per_class(y_test.values, rf_pred, rf_prob, target_cols)
    
    results_all.append({
        'Model': 'Random Forest',
        'Approach': 'Class-Weight',
        'Features': 'Top-200',
        'Accuracy': rf_macro[0],
        'Precision': rf_macro[1],
        'Recall': rf_macro[2],
        'F1_Score': rf_macro[3],
        'ROC_AUC': rf_macro[4]
    })
    
    rf_per_class['Model'] = 'Random Forest'
    per_class_results.append(rf_per_class)
    
    print(f"\nRandom Forest Results:")
    print(f"  Accuracy: {rf_macro[0]:.4f}")
    print(f"  Precision: {rf_macro[1]:.4f}")
    print(f"  Recall: {rf_macro[2]:.4f}")
    print(f"  F1 Score: {rf_macro[3]:.4f}")
    print(f"  ROC-AUC: {rf_macro[4]:.4f}")
    
    # ================= Train SVM =================
    print("\n" + "="*80)
    print("SVM MODEL (Class-Weight)")
    print("="*80)
    print("  Training SVM...")
    svm_pred, svm_prob = train_svm_class_weight(
        X_train, y_train, X_val, y_val, X_test, y_test, target_cols
    )
    
    svm_macro = evaluate_model_macro(y_test.values, svm_pred, svm_prob)
    svm_per_class = evaluate_per_class(y_test.values, svm_pred, svm_prob, target_cols)
    
    results_all.append({
        'Model': 'SVM',
        'Approach': 'Class-Weight',
        'Features': 'Top-200',
        'Accuracy': svm_macro[0],
        'Precision': svm_macro[1],
        'Recall': svm_macro[2],
        'F1_Score': svm_macro[3],
        'ROC_AUC': svm_macro[4]
    })
    
    svm_per_class['Model'] = 'SVM'
    per_class_results.append(svm_per_class)
    
    print(f"\nSVM Results:")
    print(f"  Accuracy: {svm_macro[0]:.4f}")
    print(f"  Precision: {svm_macro[1]:.4f}")
    print(f"  Recall: {svm_macro[2]:.4f}")
    print(f"  F1 Score: {svm_macro[3]:.4f}")
    print(f"  ROC-AUC: {svm_macro[4]:.4f}")
    
    # ================= Save Results =================
    results_df = pd.DataFrame(results_all)
    per_class_df = pd.concat(per_class_results, ignore_index=True)
    
    print("\n" + "="*80)
    print("OVERALL METRICS COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'rf_svm_class_weight_comparison.csv'), index=False)
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    print(per_class_df.to_string(index=False))
    per_class_df.to_csv(os.path.join(RESULTS_DIR, 'rf_svm_class_weight_per_class_metrics.csv'), index=False)
    
    # ================= Create Visualizations =================
    create_comparison_visualizations(results_df, rf_per_class, svm_per_class, target_cols)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Visualizations saved to: {REPORT_DIR}")

if __name__ == "__main__":
    main()

