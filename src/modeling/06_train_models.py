import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import time

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FEATURE_DIR = os.path.join(BASE_DIR, 'reports/feature_selection/')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/')

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_imputed.csv'), index_col='ecg_id')
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_imputed.csv'), index_col='ecg_id')
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_imputed.csv'), index_col='ecg_id')
    
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Prepare X and y
    # We need to drop non-feature columns. 
    # The feature selection script already identified numeric features.
    # We will filter columns based on the Top-N lists later.
    
    return train_df, val_df, test_df, target_cols

def load_features(n):
    path = os.path.join(FEATURE_DIR, f'top{n}_features.csv')
    df = pd.read_csv(path)
    return df['Feature'].tolist()

def evaluate_model(y_true, y_pred, y_prob, exp_name):
    # Calculate metrics
    # For multi-label, we use 'macro' average for F1
    # But wait, is this multi-label or multi-class?
    # The paper says "diagnostic superclass". A record can have multiple.
    # So it is Multi-Label Classification.
    # RandomForestClassifier supports multi-output classification natively.
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    try:
        roc_auc = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError:
        roc_auc = 0.0 # Handle edge cases
        
    print(f"[{exp_name}] Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    return {
        'Experiment': exp_name,
        'Accuracy': accuracy,
        'Macro_F1': macro_f1,
        'ROC_AUC': roc_auc
    }

def run_experiment_class_weight(train_df, test_df, features, target_cols, exp_name):
    print(f"\nRunning {exp_name}...")
    
    X_train = train_df[features]
    y_train = train_df[target_cols]
    
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    # Train RF with class_weight='balanced'
    clf = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = np.array(clf.predict_proba(X_test))
    
    # predict_proba returns a list of arrays (one for each class) for multi-output
    # We need to reshape it to (n_samples, n_classes)
    # Each element in y_prob list is (n_samples, 2) -> we want the prob of class 1
    y_prob_formatted = np.array([prob[:, 1] for prob in y_prob]).T
    
    return evaluate_model(y_test, y_pred, y_prob_formatted, exp_name)

def run_experiment_ensemble_undersampling(train_df, test_df, features, target_cols, exp_name, n_estimators=50):
    print(f"\nRunning {exp_name} (Ensemble Undersampling - {n_estimators} iterations)...")
    
    X_train_full = train_df[features]
    y_train_full = train_df[target_cols]
    
    X_test = test_df[features]
    y_test = test_df[target_cols]
    
    # Store sum of probabilities
    y_prob_sum = np.zeros((len(X_test), len(target_cols)))
    
    # Since this is multi-label, standard RandomUnderSampler doesn't work directly out of the box for the whole label set at once
    # in a way that balances ALL labels simultaneously perfectly.
    # However, a common strategy for multi-label undersampling is:
    # "LP-RUS" (Label Powerset Random Under Sampling) or simply undersampling the "majority" (usually NORM) vs others.
    # BUT, the user instruction says: "Eğitim setinde tüm sınıfları, en küçük sınıf kadar örnek içerecek şekilde RandomUnderSampler"
    # This implies treating it as a multi-class problem or balancing each binary classifier?
    # Given the context of "Ensemble", usually we train N models.
    # Let's assume we treat the problem as: For each iteration, we create a balanced subset.
    # But how to balance multi-label?
    # Simplest approximation: Treat 'NORM' as majority and everything else as minority?
    # Or use the Label Powerset approach implicitly?
    # Let's try a simpler approach often used: 
    # For each iteration, we sample a subset where the dominant class (NORM) is downsampled to match the size of other classes?
    # Actually, the user said: "Eğitim setinde tüm sınıfları, en küçük sınıf kadar örnek içerecek şekilde"
    # This strongly suggests treating it as Multi-Class (one label per record) for sampling purposes, OR
    # we have to be creative.
    # PTB-XL has 'diagnostic_superclass'. A record can have MULTIPLE.
    # But usually one is dominant or we can use the 'diagnostic_superclass' column (list of strings) 
    # and maybe convert to a single string for sampling (Label Powerset).
    
    # Let's use the 'diagnostic_superclass' column if available, or reconstruct it.
    # The cleaned file dropped it? No, we kept it in the raw loading but maybe dropped in cleaning?
    # Let's check if we can reconstruct a "primary label" for sampling.
    # If a record has MI, it's MI. If NORM, it's NORM.
    # Priority: MI > STTC > CD > HYP > NORM?
    # Let's use a custom sampler loop.
    
    # 1. Identify counts
    # We will use a simplified "Main Class" for sampling to ensure diversity.
    # If a record has multiple labels, we assign it to the rarest label it has.
    
    class_counts = y_train_full.sum().sort_values()
    min_class_name = class_counts.index[0]
    min_count = class_counts[0]
    
    print(f"Minority Class: {min_class_name} (Count: {min_count})")
    
    # Assign a single 'sampling_label' to each record based on rarity
    # Rarity order: HYP (lowest usually) -> ... -> NORM (highest)
    # Actually let's just use the counts we just found.
    sorted_classes = class_counts.index.tolist() # ['HYP', 'CD', 'STTC', 'MI', 'NORM'] (example)
    
    def get_sampling_label(row):
        for cls in sorted_classes:
            if row[cls] == 1:
                return cls
        return 'NORM' # Default/Fallback
    
    sampling_labels = y_train_full.apply(get_sampling_label, axis=1)
    
    start_time = time.time()
    
    for i in range(n_estimators):
        # Create a balanced subset
        # We want 'min_count' samples from EACH class defined in 'sampling_labels'
        indices_to_keep = []
        for cls in sorted_classes:
            cls_indices = sampling_labels[sampling_labels == cls].index
            if len(cls_indices) >= min_count:
                selected = np.random.choice(cls_indices, min_count, replace=False)
            else:
                # Should not happen if min_count is the minimum, but for safety
                selected = cls_indices 
            indices_to_keep.extend(selected)
            
        X_resampled = X_train_full.loc[indices_to_keep]
        y_resampled = y_train_full.loc[indices_to_keep]
        
        # Train Model
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=i, n_jobs=-1) # Smaller RF for ensemble
        clf.fit(X_resampled, y_resampled)
        
        # Predict Probabilities
        probs = np.array(clf.predict_proba(X_test))
        # probs is list of (n_samples, 2) arrays
        probs_formatted = np.array([p[:, 1] for p in probs]).T
        
        y_prob_sum += probs_formatted
        
    elapsed = time.time() - start_time
    print(f"Ensemble training took {elapsed:.2f} seconds.")
    
    # Average Probabilities
    y_prob_avg = y_prob_sum / n_estimators
    
    # Thresholding (0.5 for now, can be tuned on Val set but we stick to standard for comparison)
    y_pred_avg = (y_prob_avg >= 0.5).astype(int)
    
    return evaluate_model(y_test, y_pred_avg, y_prob_avg, exp_name)

def main():
    train_df, val_df, test_df, target_cols = load_data()
    
    results = []
    
    # Exp 1: RF + Top-50 + class_weight
    features_50 = load_features(50)
    res1 = run_experiment_class_weight(train_df, test_df, features_50, target_cols, "RF_Top50_ClassWeight")
    results.append(res1)
    
    # Exp 2: RF + Top-100 + class_weight
    features_100 = load_features(100)
    res2 = run_experiment_class_weight(train_df, test_df, features_100, target_cols, "RF_Top100_ClassWeight")
    results.append(res2)
    
    # Exp 3: RF + Top-200 + class_weight
    features_200 = load_features(200)
    res3 = run_experiment_class_weight(train_df, test_df, features_200, target_cols, "RF_Top200_ClassWeight")
    results.append(res3)
    
    # Exp 4: RF + Top-200 + Ensemble Undersampling
    res4 = run_experiment_ensemble_undersampling(train_df, test_df, features_200, target_cols, "RF_Top200_Ensemble")
    results.append(res4)
    
    # Save Results
    results_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(results_df)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_results.csv'), index=False)

if __name__ == "__main__":
    main()
