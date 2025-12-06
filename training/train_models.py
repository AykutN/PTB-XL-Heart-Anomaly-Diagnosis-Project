"""
Model Training and Evaluation for ECG Classification

This script trains and evaluates three classifiers:
1. Decision Tree
2. Gaussian Naive Bayes
3. Support Vector Machine (SVM)

The models are trained on the preprocessed PTB-XL+ features and evaluated
on the test set. Results are saved for inclusion in the scientific report.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SELECTED_DIR = os.path.join(DATA_DIR, 'selected')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class names
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def load_data(feature_set='top100'):
    """Load preprocessed data for a specific feature set."""
    print(f"Loading {feature_set} feature set...")
    
    if feature_set == 'full':
        X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
        X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
        X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    else:
        # Load selected features
        suffix = feature_set.replace('top', '_top')
        X_train = pd.read_csv(os.path.join(SELECTED_DIR, f'X_train{suffix}.csv'))
        X_val = pd.read_csv(os.path.join(SELECTED_DIR, f'X_val{suffix}.csv'))
        X_test = pd.read_csv(os.path.join(SELECTED_DIR, f'X_test{suffix}.csv'))
    
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))['label']
    y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv'))['label']
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))['label']
    
    # Load class weights
    class_weights_df = pd.read_csv(os.path.join(DATA_DIR, 'class_weights.csv'))
    class_weights = dict(zip(class_weights_df.columns, class_weights_df.iloc[0].values))
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


def get_models_and_params(class_weights):
    """Define models and their hyperparameter grids."""
    
    models = {
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42, class_weight=class_weights),
            'params': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy'],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'NaiveBayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, class_weight=class_weights, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
    }
    
    return models


def train_model_with_grid_search(model_name, model_config, X_train, y_train, X_val, y_val):
    """Train a model using grid search for hyperparameter tuning."""
    
    print(f"\nTraining {model_name}...")
    
    # Combine train and validation for cross-validation
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    
    # Create custom CV that separates train and validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search
    start_time = time.time()
    
    grid_search = GridSearchCV(
        model_config['model'],
        model_config['params'],
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    
    grid_search.fit(X_combined, y_combined)
    
    training_time = time.time() - start_time
    
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score (macro F1): {grid_search.best_score_:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model on the test set."""
    
    print(f"\nEvaluating {model_name} on test set...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probability predictions for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
        'macro_precision': precision_score(y_test, y_pred, average='macro'),
        'macro_recall': recall_score(y_test, y_pred, average='macro'),
    }
    
    # Per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    
    # ROC-AUC (one-vs-rest)
    if y_prob is not None:
        try:
            # Binarize labels for multi-class ROC-AUC
            y_test_bin = label_binarize(y_test, classes=CLASS_NAMES)
            
            # Handle the case where we might not have all classes
            roc_auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
            metrics['roc_auc'] = roc_auc
        except Exception as e:
            print(f"  Warning: Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    # Print summary
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    if metrics['roc_auc']:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics, class_report, cm, y_pred, y_prob


def plot_confusion_matrix(cm, model_name, feature_set, save_path):
    """Plot and save confusion matrix."""
    
    plt.figure(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1)
    
    # Add raw counts as secondary annotation
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j + 0.5, i + 0.75, f'({cm[i, j]})',
                    ha='center', va='center', fontsize=8, color='gray')
    
    plt.title(f'Confusion Matrix: {model_name} ({feature_set})\n(normalized, with counts)',
              fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_roc_curves(y_test, y_prob, model_name, feature_set, save_path):
    """Plot ROC curves for each class."""
    
    if y_prob is None:
        return
    
    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=CLASS_NAMES)
    n_classes = len(CLASS_NAMES)
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves: {model_name} ({feature_set})', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def save_results(all_results, feature_set):
    """Save results to CSV for report."""
    
    # Summary table
    summary_data = []
    for model_name, result in all_results.items():
        row = {
            'Model': model_name,
            'Feature Set': feature_set,
            'Accuracy': result['metrics']['accuracy'],
            'Macro F1': result['metrics']['macro_f1'],
            'Weighted F1': result['metrics']['weighted_f1'],
            'Macro Precision': result['metrics']['macro_precision'],
            'Macro Recall': result['metrics']['macro_recall'],
            'ROC-AUC': result['metrics'].get('roc_auc'),
            'Training Time (s)': result['training_time']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(RESULTS_DIR, f'summary_{feature_set}.csv'), index=False)
    
    # Per-class metrics
    for model_name, result in all_results.items():
        class_report = result['class_report']
        class_data = []
        for class_name in CLASS_NAMES:
            if class_name in class_report:
                row = {
                    'Class': class_name,
                    'Precision': class_report[class_name]['precision'],
                    'Recall': class_report[class_name]['recall'],
                    'F1-Score': class_report[class_name]['f1-score'],
                    'Support': class_report[class_name]['support']
                }
                class_data.append(row)
        
        class_df = pd.DataFrame(class_data)
        class_df.to_csv(
            os.path.join(RESULTS_DIR, f'class_metrics_{model_name}_{feature_set}.csv'),
            index=False
        )
    
    print(f"\nResults saved to {RESULTS_DIR}")


def create_comparison_plots(all_results_by_feature):
    """Create comparison plots across models and feature sets."""
    
    # Prepare data
    comparison_data = []
    for feature_set, results in all_results_by_feature.items():
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Feature Set': feature_set,
                'Accuracy': result['metrics']['accuracy'],
                'Macro F1': result['metrics']['macro_f1'],
                'Training Time': result['training_time']
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Plot 1: Macro F1 comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart for Macro F1
    ax1 = axes[0]
    x = np.arange(len(['top50', 'top100', 'top200']))
    width = 0.25
    
    for i, model in enumerate(['DecisionTree', 'NaiveBayes', 'SVM']):
        model_data = df[df['Model'] == model].sort_values('Feature Set')
        bars = ax1.bar(x + i*width, model_data['Macro F1'], width, label=model, alpha=0.8)
    
    ax1.set_xlabel('Feature Set')
    ax1.set_ylabel('Macro F1 Score')
    ax1.set_title('Model Comparison: Macro F1 Score', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['Top 50', 'Top 100', 'Top 200'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Training time comparison
    ax2 = axes[1]
    
    for i, model in enumerate(['DecisionTree', 'NaiveBayes', 'SVM']):
        model_data = df[df['Model'] == model].sort_values('Feature Set')
        bars = ax2.bar(x + i*width, model_data['Training Time'], width, label=model, alpha=0.8)
    
    ax2.set_xlabel('Feature Set')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Model Comparison: Training Time', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(['Top 50', 'Top 100', 'Top 200'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved model comparison plots")


def main():
    """Main training and evaluation pipeline."""
    
    print("="*60)
    print("ECG Classification Model Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Feature sets to evaluate
    feature_sets = ['top50', 'top100', 'top200']
    
    all_results_by_feature = {}
    
    for feature_set in feature_sets:
        print(f"\n{'='*60}")
        print(f"Processing feature set: {feature_set.upper()}")
        print("="*60)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_data(feature_set)
        
        # Get models
        models = get_models_and_params(class_weights)
        
        all_results = {}
        
        for model_name, model_config in models.items():
            # Train model
            trained_model, best_params, training_time = train_model_with_grid_search(
                model_name, model_config, X_train, y_train, X_val, y_val
            )
            
            # Evaluate model
            metrics, class_report, cm, y_pred, y_prob = evaluate_model(
                trained_model, X_test, y_test, model_name
            )
            
            # Store results
            all_results[model_name] = {
                'model': trained_model,
                'best_params': best_params,
                'training_time': training_time,
                'metrics': metrics,
                'class_report': class_report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            # Plot confusion matrix
            cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}_{feature_set}.png')
            plot_confusion_matrix(cm, model_name, feature_set, cm_path)
            
            # Plot ROC curves
            roc_path = os.path.join(RESULTS_DIR, f'roc_curves_{model_name}_{feature_set}.png')
            plot_roc_curves(y_test, y_prob, model_name, feature_set, roc_path)
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f'{model_name}_{feature_set}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(trained_model, f)
            print(f"  Model saved to {model_path}")
        
        # Save results for this feature set
        save_results(all_results, feature_set)
        all_results_by_feature[feature_set] = all_results
    
    # Create comparison plots
    create_comparison_plots(all_results_by_feature)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for feature_set, results in all_results_by_feature.items():
        print(f"\n{feature_set.upper()}:")
        for model_name, result in results.items():
            print(f"  {model_name}: Macro F1 = {result['metrics']['macro_f1']:.4f}, "
                  f"Accuracy = {result['metrics']['accuracy']:.4f}")
    
    # Find best model
    best_score = 0
    best_model = None
    best_feature = None
    
    for feature_set, results in all_results_by_feature.items():
        for model_name, result in results.items():
            if result['metrics']['macro_f1'] > best_score:
                best_score = result['metrics']['macro_f1']
                best_model = model_name
                best_feature = feature_set
    
    print(f"\n*** Best model: {best_model} with {best_feature} features ***")
    print(f"*** Macro F1 Score: {best_score:.4f} ***")
    
    print("\n" + "="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
