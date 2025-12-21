import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Define paths
BASE_DIR = "/Users/y.aykut/Codebase/PTB-XL+ paper/"
DATA_DIR = os.path.join(BASE_DIR, "data/processed/")
FEATURE_DIR = os.path.join(BASE_DIR, "reports/feature_selection/")
RESULTS_DIR = os.path.join(BASE_DIR, "results/")

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_imputed.csv"), index_col="ecg_id")
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_imputed.csv"), index_col="ecg_id")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_imputed.csv"), index_col="ecg_id")
    target_cols = ["NORM", "MI", "STTC", "CD", "HYP"]
    return train_df, val_df, test_df, target_cols


def load_features(n):
    path = os.path.join(FEATURE_DIR, f"top{n}_features.csv")
    df = pd.read_csv(path)
    return df["Feature"].tolist()


def evaluate_model(y_true, y_pred, y_prob, exp_name):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        roc_auc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        roc_auc = 0.0

    print(f"[{exp_name}] Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "Experiment": exp_name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "ROC_AUC": roc_auc,
    }


def run_experiment_class_weight(train_df, test_df, features, target_cols, exp_name):
    print(f"\nRunning {exp_name}...")

    X_train = train_df[features]
    y_train = train_df[target_cols]

    X_test = test_df[features]
    y_test = test_df[target_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = OneVsRestClassifier(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    )
    clf.fit(X_train_scaled, y_train)

    y_prob = clf.predict_proba(X_test_scaled)
    y_pred = (y_prob >= 0.5).astype(int)

    return evaluate_model(y_test, y_pred, y_prob, exp_name)


def run_experiment_ensemble_undersampling(
    train_df, test_df, features, target_cols, exp_name, n_estimators=50
):
    print(f"\nRunning {exp_name} (Ensemble Undersampling - {n_estimators} iterations)...")

    X_train_full = train_df[features]
    y_train_full = train_df[target_cols]

    X_test = test_df[features]
    y_test = test_df[target_cols]

    y_prob_sum = np.zeros((len(X_test), len(target_cols)))

    class_counts = y_train_full.sum().sort_values()
    min_count = int(class_counts.iloc[0])
    sorted_classes = class_counts.index.tolist()

    def get_sampling_label(row):
        for cls in sorted_classes:
            if row[cls] == 1:
                return cls
        return sorted_classes[-1]

    sampling_labels = y_train_full.apply(get_sampling_label, axis=1)

    start_time = time.time()

    for i in range(n_estimators):
        indices_to_keep = []
        for cls in sorted_classes:
            cls_indices = sampling_labels[sampling_labels == cls].index
            if len(cls_indices) >= min_count:
                selected = np.random.choice(cls_indices, min_count, replace=False)
            else:
                selected = cls_indices
            indices_to_keep.extend(selected)

        X_resampled = X_train_full.loc[indices_to_keep]
        y_resampled = y_train_full.loc[indices_to_keep]

        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)
        X_test_scaled = scaler.transform(X_test)

        clf = OneVsRestClassifier(
            SVC(kernel="rbf", probability=True, class_weight=None, random_state=i)
        )
        clf.fit(X_resampled_scaled, y_resampled)

        probs = clf.predict_proba(X_test_scaled)
        y_prob_sum += probs

    elapsed = time.time() - start_time
    print(f"Ensemble training took {elapsed:.2f} seconds.")

    y_prob_avg = y_prob_sum / n_estimators
    y_pred_avg = (y_prob_avg >= 0.5).astype(int)

    return evaluate_model(y_test, y_pred_avg, y_prob_avg, exp_name)


def main():
    train_df, val_df, test_df, target_cols = load_data()

    results = []

    features_50 = load_features(50)
    res1 = run_experiment_class_weight(
        train_df, test_df, features_50, target_cols, "SVM_Top50_ClassWeight"
    )
    results.append(res1)

    features_100 = load_features(100)
    res2 = run_experiment_class_weight(
        train_df, test_df, features_100, target_cols, "SVM_Top100_ClassWeight"
    )
    results.append(res2)

    features_200 = load_features(200)
    res3 = run_experiment_class_weight(
        train_df, test_df, features_200, target_cols, "SVM_Top200_ClassWeight"
    )
    results.append(res3)

    res4 = run_experiment_ensemble_undersampling(
        train_df, test_df, features_200, target_cols, "SVM_Top200_Ensemble"
    )
    results.append(res4)

    results_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(results_df)
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "svm_model_comparison_results.csv"), index=False
    )


if __name__ == "__main__":
    main()
