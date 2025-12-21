import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import time

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FEATURE_DIR = os.path.join(BASE_DIR, 'reports/feature_selection/')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/')

def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_imputed.csv'), index_col='ecg_id')
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_imputed.csv'), index_col='ecg_id')
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    return train_df, test_df, target_cols

def load_features(n):
    path = os.path.join(FEATURE_DIR, f'top{n}_features.csv')
    df = pd.read_csv(path)
    return df['Feature'].tolist()

def generate_per_class_report(y_true, y_pred, target_cols, model_name):
    report_dict = classification_report(y_true, y_pred, target_names=target_cols, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Filter for classes only (exclude micro/macro avg for the main table)
    class_metrics = report_df.loc[target_cols]
    class_metrics['Model'] = model_name
    return class_metrics

def run_best_models_detailed():
    train_df, test_df, target_cols = load_data()
    features = load_features(200) # Using Top-200 as it was best
    
    X_train = train_df[features]
    y_train = train_df[target_cols]
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    # 1. RF Class Weight
    print("Training RF Class Weight (Detailed)...")
    clf_cw = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_cw.fit(X_train, y_train)
    y_pred_cw = clf_cw.predict(X_test)
    
    metrics_cw = generate_per_class_report(y_test, y_pred_cw, target_cols, "RF_ClassWeight")
    
    # 2. RF Ensemble Undersampling
    print("Training RF Ensemble (Detailed)...")
    # Re-implementing the ensemble logic briefly to get predictions
    n_estimators = 50
    y_prob_sum = np.zeros((len(X_test), len(target_cols)))
    
    class_counts = y_train.sum().sort_values()
    min_count = class_counts.iloc[0]
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
        
        probs = np.array(clf.predict_proba(X_test))
        probs_formatted = np.array([p[:, 1] for p in probs]).T
        y_prob_sum += probs_formatted
        
    y_prob_avg = y_prob_sum / n_estimators
    y_pred_ens = (y_prob_avg >= 0.5).astype(int)
    
    metrics_ens = generate_per_class_report(y_test, y_pred_ens, target_cols, "RF_Ensemble")
    
    # Combine and Save
    combined_metrics = pd.concat([metrics_cw, metrics_ens])
    print("\nCombined Per-Class Metrics:")
    print(combined_metrics)
    
    combined_metrics.to_csv(os.path.join(RESULTS_DIR, 'per_class_metrics_comparison.csv'))
    print(f"Saved detailed metrics to {os.path.join(RESULTS_DIR, 'per_class_metrics_comparison.csv')}")

if __name__ == "__main__":
    run_best_models_detailed()
