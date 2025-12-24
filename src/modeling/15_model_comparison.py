"""
Comprehensive Model Comparison: Random Forest vs SVM vs Naive Bayes vs Logistic Regression
Includes detailed metrics, per-class analysis, and visualizations for paper.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
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

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, target_cols):
    """Train Random Forest model with threshold optimization."""
    print("  Training Random Forest...")
    
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
    
    return clf, y_pred, y_test_prob, thresholds

def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, target_cols):
    """Train SVM model with threshold optimization."""
    print("  Training SVM (scaling data first)...")
    
    # Scale data for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Use MultiOutputClassifier for multi-label classification
    base_svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42, cache_size=1000)
    clf = MultiOutputClassifier(base_svm, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # Get probabilities
    y_val_prob = np.array([est.predict_proba(X_val_scaled)[:, 1] for est in clf.estimators_]).T
    y_test_prob = np.array([est.predict_proba(X_test_scaled)[:, 1] for est in clf.estimators_]).T
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
    y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
    
    return clf, y_pred, y_test_prob, thresholds, scaler

def train_naive_bayes(X_train, y_train, X_val, y_val, X_test, y_test, target_cols):
    """Train Naive Bayes model with threshold optimization."""
    print("  Training Naive Bayes...")
    
    # Use MultiOutputClassifier for multi-label classification
    base_nb = GaussianNB()
    clf = MultiOutputClassifier(base_nb, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Get probabilities
    y_val_prob = np.array([est.predict_proba(X_val)[:, 1] for est in clf.estimators_]).T
    y_test_prob = np.array([est.predict_proba(X_test)[:, 1] for est in clf.estimators_]).T
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
    y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
    
    return clf, y_pred, y_test_prob, thresholds

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, target_cols):
    """Train Logistic Regression model with threshold optimization."""
    print("  Training Logistic Regression...")
    
    # Scale data for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Use MultiOutputClassifier for multi-label classification
    base_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
    clf = MultiOutputClassifier(base_lr, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # Get probabilities
    y_val_prob = np.array([est.predict_proba(X_val_scaled)[:, 1] for est in clf.estimators_]).T
    y_test_prob = np.array([est.predict_proba(X_test_scaled)[:, 1] for est in clf.estimators_]).T
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(y_val, y_val_prob, target_cols)
    y_pred = apply_thresholds(y_test_prob, thresholds, target_cols)
    
    return clf, y_pred, y_test_prob, thresholds, scaler

def create_comparison_visualizations(results_df, per_class_all, target_cols):
    """Create comprehensive visualization plots."""
    print("\nCreating visualizations...")
    
    # 1. Overall metrics comparison bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    model_colors = {
        'Random Forest': '#3498db',
        'SVM': '#e74c3c',
        'Naive Bayes': '#2ecc71',
        'Logistic Regression': '#f39c12'
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        x = np.arange(len(results_df))
        width = 0.2
        
        for i, (model_name, color) in enumerate(model_colors.items()):
            model_values = results_df[results_df['Model'] == model_name][metric].values
            if len(model_values) > 0:
                bars = ax.bar(x + i*width - width*1.5, model_values[0], width,
                             label=model_name, color=color, alpha=0.8)
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels(['Models'], rotation=0)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.suptitle('Model Performance Comparison: All Models', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'model_comparison_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved metrics comparison to {os.path.join(REPORT_DIR, 'model_comparison_metrics.png')}")
    plt.close()
    
    # 2. Per-class F1 score comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(target_cols))
    width = 0.2
    
    models_list = ['Random Forest', 'SVM', 'Naive Bayes', 'Logistic Regression']
    for i, model_name in enumerate(models_list):
        model_data = per_class_all[per_class_all['Model'] == model_name]
        if len(model_data) > 0:
            bars = ax.bar(x + i*width - width*1.5, model_data['F1_Score'].values, width,
                         label=model_name, color=model_colors[model_name], alpha=0.8)
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Diagnostic Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(target_cols)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'per_class_f1_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved per-class F1 comparison to {os.path.join(REPORT_DIR, 'per_class_f1_comparison.png')}")
    plt.close()
    
    # 3. Per-class metrics heatmap for each model
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models_list):
        model_data = per_class_all[per_class_all['Model'] == model_name]
        if len(model_data) > 0:
            metrics_df = model_data.set_index('Class')[['Precision', 'Recall', 'F1_Score', 'ROC_AUC']]
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                       vmin=0, vmax=1, ax=axes[idx], cbar_kws={'label': 'Score'})
            axes[idx].set_title(f'{model_name} - Per-Class Metrics', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Diagnostic Class', fontsize=10)
    
    plt.suptitle('Per-Class Performance Metrics Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'per_class_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved metrics heatmap to {os.path.join(REPORT_DIR, 'per_class_metrics_heatmap.png')}")
    plt.close()

def main():
    """Main function to run model comparison."""
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON: RF vs SVM vs Naive Bayes vs Logistic Regression")
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
    print("RANDOM FOREST MODEL")
    print("="*80)
    rf_model, rf_pred, rf_prob, rf_thresholds = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test, target_cols
    )
    
    rf_macro = evaluate_model_macro(y_test.values, rf_pred, rf_prob)
    rf_per_class = evaluate_per_class(y_test.values, rf_pred, rf_prob, target_cols)
    
    results_all.append({
        'Model': 'Random Forest',
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
    print("SVM MODEL")
    print("="*80)
    svm_model, svm_pred, svm_prob, svm_thresholds, svm_scaler = train_svm(
        X_train, y_train, X_val, y_val, X_test, y_test, target_cols
    )
    
    svm_macro = evaluate_model_macro(y_test.values, svm_pred, svm_prob)
    svm_per_class = evaluate_per_class(y_test.values, svm_pred, svm_prob, target_cols)
    
    results_all.append({
        'Model': 'SVM',
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
    
    # ================= Train Naive Bayes =================
    print("\n" + "="*80)
    print("NAIVE BAYES MODEL")
    print("="*80)
    nb_model, nb_pred, nb_prob, nb_thresholds = train_naive_bayes(
        X_train, y_train, X_val, y_val, X_test, y_test, target_cols
    )
    
    nb_macro = evaluate_model_macro(y_test.values, nb_pred, nb_prob)
    nb_per_class = evaluate_per_class(y_test.values, nb_pred, nb_prob, target_cols)
    
    results_all.append({
        'Model': 'Naive Bayes',
        'Accuracy': nb_macro[0],
        'Precision': nb_macro[1],
        'Recall': nb_macro[2],
        'F1_Score': nb_macro[3],
        'ROC_AUC': nb_macro[4]
    })
    
    nb_per_class['Model'] = 'Naive Bayes'
    per_class_results.append(nb_per_class)
    
    print(f"\nNaive Bayes Results:")
    print(f"  Accuracy: {nb_macro[0]:.4f}")
    print(f"  Precision: {nb_macro[1]:.4f}")
    print(f"  Recall: {nb_macro[2]:.4f}")
    print(f"  F1 Score: {nb_macro[3]:.4f}")
    print(f"  ROC-AUC: {nb_macro[4]:.4f}")
    
    # ================= Train Logistic Regression =================
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION MODEL")
    print("="*80)
    lr_model, lr_pred, lr_prob, lr_thresholds, lr_scaler = train_logistic_regression(
        X_train, y_train, X_val, y_val, X_test, y_test, target_cols
    )
    
    lr_macro = evaluate_model_macro(y_test.values, lr_pred, lr_prob)
    lr_per_class = evaluate_per_class(y_test.values, lr_pred, lr_prob, target_cols)
    
    results_all.append({
        'Model': 'Logistic Regression',
        'Accuracy': lr_macro[0],
        'Precision': lr_macro[1],
        'Recall': lr_macro[2],
        'F1_Score': lr_macro[3],
        'ROC_AUC': lr_macro[4]
    })
    
    lr_per_class['Model'] = 'Logistic Regression'
    per_class_results.append(lr_per_class)
    
    print(f"\nLogistic Regression Results:")
    print(f"  Accuracy: {lr_macro[0]:.4f}")
    print(f"  Precision: {lr_macro[1]:.4f}")
    print(f"  Recall: {lr_macro[2]:.4f}")
    print(f"  F1 Score: {lr_macro[3]:.4f}")
    print(f"  ROC-AUC: {lr_macro[4]:.4f}")
    
    # ================= Save Results =================
    results_df = pd.DataFrame(results_all)
    per_class_df = pd.concat(per_class_results, ignore_index=True)
    
    print("\n" + "="*80)
    print("OVERALL METRICS COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_all.csv'), index=False)
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    print(per_class_df.to_string(index=False))
    per_class_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_per_class_metrics.csv'), index=False)
    
    # ================= Create Visualizations =================
    create_comparison_visualizations(results_df, per_class_df, target_cols)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Visualizations saved to: {REPORT_DIR}")

if __name__ == "__main__":
    main()

