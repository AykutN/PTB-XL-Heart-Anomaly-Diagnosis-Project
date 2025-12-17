import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

# =============================================================================
# STEP 1: Load Preprocessed Data
# =============================================================================

print("="*60)
print("STEP 1: Loading Preprocessed Data")
print("="*60)

X_train = pd.read_csv("../data/processed/X_train.csv")
X_val = pd.read_csv("../data/processed/X_val.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")

y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
y_val = pd.read_csv("../data/processed/y_val.csv").squeeze()
y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

feature_names = pd.read_csv("../data/processed/feature_names.csv")["feature"].tolist()

print(f"  X_train shape: {X_train.shape}")
print(f"  Features: {len(feature_names)}")

# =============================================================================
# STEP 2: Method 1 - Variance Threshold
# =============================================================================

print("\n" + "="*60)
print("STEP 2: Variance Threshold")
print("="*60)

# Remove features with very low variance (near-constant)
# Threshold: variance < 0.01 means feature is almost constant

selector_var = VarianceThreshold(threshold=0.01)
selector_var.fit(X_train)

low_variance_mask = selector_var.get_support()
n_removed = (~low_variance_mask).sum()

print(f"  Original features: {X_train.shape[1]}")
print(f"  Low variance features removed: {n_removed}")
print(f"  Remaining features: {low_variance_mask.sum()}")

# Show some removed features
removed_features = [f for f, keep in zip(feature_names, low_variance_mask) if not keep]
if removed_features:
    print(f"  Example removed features: {removed_features[:5]}")

# =============================================================================
# STEP 3: Method 2 - Correlation Filter
# =============================================================================

print("\n" + "="*60)
print("STEP 3: Correlation Filter")
print("="*60)

# Remove highly correlated features (redundant information)
# If two features have correlation > 0.95, keep only one

# Apply variance filter first
X_train_var = X_train.loc[:, low_variance_mask]
feature_names_var = [f for f, keep in zip(feature_names, low_variance_mask) if keep]

# Calculate correlation matrix
corr_matrix = X_train_var.corr().abs()

# Find highly correlated pairs
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_features = set()
threshold = 0.95

for column in upper_triangle.columns:
    correlated = upper_triangle.index[upper_triangle[column] > threshold].tolist()
    if correlated:
        high_corr_features.add(column)

print(f"  Correlation threshold: {threshold}")
print(f"  Highly correlated features to remove: {len(high_corr_features)}")
print(f"  Remaining features: {len(feature_names_var) - len(high_corr_features)}")

# Keep features that are not highly correlated
keep_features_corr = [f for f in feature_names_var if f not in high_corr_features]

# =============================================================================
# STEP 4: Method 3 - Mutual Information
# =============================================================================

print("\n" + "="*60)
print("STEP 4: Mutual Information (Feature Importance)")
print("="*60)

# Mutual Information measures dependency between feature and target
# Higher = more informative

X_train_filtered = X_train[keep_features_corr]

# Calculate MI scores (this takes a moment)
print("  Calculating MI scores (this may take a minute)...")
mi_scores = mutual_info_classif(X_train_filtered, y_train, random_state=42)

mi_df = pd.DataFrame({
    "feature": keep_features_corr,
    "mi_score": mi_scores
}).sort_values("mi_score", ascending=False)

print(f"\n  Top 20 features by Mutual Information:")
print(mi_df.head(20).to_string(index=False))

# =============================================================================
# STEP 5: Method 4 - Tree-based Feature Importance
# =============================================================================

print("\n" + "="*60)
print("STEP 5: Random Forest Feature Importance")
print("="*60)

# Train a simple RF to get feature importances
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_filtered, y_train)

rf_importance = pd.DataFrame({
    "feature": keep_features_corr,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print(f"\n  Top 20 features by Random Forest Importance:")
print(rf_importance.head(20).to_string(index=False))

# =============================================================================
# STEP 6: Combine Rankings
# =============================================================================

print("\n" + "="*60)
print("STEP 6: Combined Feature Ranking")
print("="*60)

# Merge MI and RF rankings
combined = mi_df.merge(rf_importance, on="feature")
combined["mi_rank"] = combined["mi_score"].rank(ascending=False)
combined["rf_rank"] = combined["importance"].rank(ascending=False)
combined["avg_rank"] = (combined["mi_rank"] + combined["rf_rank"]) / 2
combined = combined.sort_values("avg_rank")

print(f"\n  Top 30 features (combined ranking):")
print(combined.head(30)[["feature", "mi_score", "importance", "avg_rank"]].to_string(index=False))

# =============================================================================
# STEP 7: Select Top K Features
# =============================================================================

print("\n" + "="*60)
print("STEP 7: Selecting Top Features")
print("="*60)

# Select top features based on different thresholds
top_50 = combined.head(50)["feature"].tolist()
top_100 = combined.head(100)["feature"].tolist()
top_200 = combined.head(200)["feature"].tolist()

print(f"  Top 50 features selected")
print(f"  Top 100 features selected")
print(f"  Top 200 features selected")

# =============================================================================
# STEP 8: PCA Analysis (Optional Dimensionality Reduction)
# =============================================================================

print("\n" + "="*60)
print("STEP 8: PCA Analysis")
print("="*60)

# Apply PCA to see how many components explain variance
pca_full = PCA()
pca_full.fit(X_train_filtered)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components for different variance thresholds
n_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_99 = np.argmax(cumulative_variance >= 0.99) + 1

print(f"  Components for 90% variance: {n_90}")
print(f"  Components for 95% variance: {n_95}")
print(f"  Components for 99% variance: {n_99}")

# =============================================================================
# STEP 9: Visualization
# =============================================================================

print("\n" + "="*60)
print("STEP 9: Creating Visualizations")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feature Selection Analysis", fontsize=16, fontweight="bold")

# 1. Top 20 MI scores
ax1 = axes[0, 0]
top20_mi = mi_df.head(20)
ax1.barh(range(20), top20_mi["mi_score"].values, color="steelblue")
ax1.set_yticks(range(20))
ax1.set_yticklabels(top20_mi["feature"].values, fontsize=8)
ax1.set_xlabel("Mutual Information Score")
ax1.set_title("Top 20 Features (Mutual Information)")
ax1.invert_yaxis()

# 2. Top 20 RF importance
ax2 = axes[0, 1]
top20_rf = rf_importance.head(20)
ax2.barh(range(20), top20_rf["importance"].values, color="coral")
ax2.set_yticks(range(20))
ax2.set_yticklabels(top20_rf["feature"].values, fontsize=8)
ax2.set_xlabel("Feature Importance")
ax2.set_title("Top 20 Features (Random Forest)")
ax2.invert_yaxis()

# 3. PCA Cumulative Variance
ax3 = axes[1, 0]
ax3.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, "b-", linewidth=2)
ax3.axhline(y=0.90, color="r", linestyle="--", label="90% variance")
ax3.axhline(y=0.95, color="g", linestyle="--", label="95% variance")
ax3.axvline(x=n_90, color="r", linestyle=":", alpha=0.5)
ax3.axvline(x=n_95, color="g", linestyle=":", alpha=0.5)
ax3.set_xlabel("Number of Components")
ax3.set_ylabel("Cumulative Explained Variance")
ax3.set_title("PCA: Cumulative Variance Explained")
ax3.legend()
ax3.set_xlim(0, 200)

# 4. Feature count summary
ax4 = axes[1, 1]
stages = ["Original", "Variance\nFilter", "Correlation\nFilter", "Top 200", "Top 100", "Top 50"]
counts = [len(feature_names), low_variance_mask.sum(), len(keep_features_corr), 200, 100, 50]
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(stages)))
ax4.bar(stages, counts, color=colors)
ax4.set_ylabel("Number of Features")
ax4.set_title("Feature Reduction Pipeline")
for i, v in enumerate(counts):
    ax4.text(i, v + 10, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("feature_selection_analysis.png", dpi=150)
plt.show()

print("  Saved: feature_selection_analysis.png")

# =============================================================================
# STEP 10: Save Selected Features
# =============================================================================

print("\n" + "="*60)
print("STEP 10: Saving Selected Features")
print("="*60)

import os
os.makedirs("../data/processed/selected", exist_ok=True)

# Save different feature sets
for name, features in [("top50", top_50), ("top100", top_100), ("top200", top_200)]:
    X_train_sel = X_train[features]
    X_val_sel = X_val[features]
    X_test_sel = X_test[features]
    
    X_train_sel.to_csv(f"../data/processed/selected/X_train_{name}.csv", index=False)
    X_val_sel.to_csv(f"../data/processed/selected/X_val_{name}.csv", index=False)
    X_test_sel.to_csv(f"../data/processed/selected/X_test_{name}.csv", index=False)
    
    pd.DataFrame({"feature": features}).to_csv(
        f"../data/processed/selected/features_{name}.csv", index=False
    )
    
    print(f"  Saved {name}: {len(features)} features")

# Save combined ranking
combined.to_csv("../data/processed/selected/feature_ranking.csv", index=False)
print("  Saved: feature_ranking.csv")

# =============================================================================
# STEP 11: Summary
# =============================================================================

print("\n" + "="*60)
print("FEATURE SELECTION COMPLETE - SUMMARY")
print("="*60)
print(f"""
Feature Reduction Pipeline:
  1. Original features: {len(feature_names)}
  2. After variance filter: {low_variance_mask.sum()} (removed {n_removed} low-variance)
  3. After correlation filter: {len(keep_features_corr)} (removed {len(high_corr_features)} redundant)
  4. Ranking methods: Mutual Information + Random Forest Importance
  5. Selected sets: Top 50, Top 100, Top 200

PCA Analysis:
  - 90% variance: {n_90} components
  - 95% variance: {n_95} components
  - 99% variance: {n_99} components

Top 10 Most Important Features:
{combined.head(10)[['feature', 'avg_rank']].to_string(index=False)}

Files Saved:
  - feature_selection_analysis.png
  - data/processed/selected/X_train_top50.csv (etc.)
  - data/processed/selected/feature_ranking.csv

Recommendation:
  - Start with Top 100 features for baseline model
  - If overfitting: reduce to Top 50
  - If underfitting: try Top 200 or PCA
""")

# =============================================================================
# OVERSAMPLED FEATURE SELECTION
# =============================================================================

print("\n" + "="*60)
print("OVERSAMPLED DATA: Loading Preprocessed Data")
print("="*60)

# Load oversampled training data (validation and test remain the same)
X_train_oversampled = pd.read_csv("../data/processed/X_train_oversampled.csv")
y_train_oversampled = pd.read_csv("../data/processed/y_train_oversampled.csv").squeeze()

print(f"  X_train_oversampled shape: {X_train_oversampled.shape}")
print(f"  Features: {len(feature_names)}")

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 2: Variance Threshold
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 2: Variance Threshold")
print("="*60)

selector_var_over = VarianceThreshold(threshold=0.01)
selector_var_over.fit(X_train_oversampled)

low_variance_mask_over = selector_var_over.get_support()
n_removed_over = (~low_variance_mask_over).sum()

print(f"  Original features: {X_train_oversampled.shape[1]}")
print(f"  Low variance features removed (oversampled): {n_removed_over}")
print(f"  Remaining features (oversampled): {low_variance_mask_over.sum()}")

removed_features_over = [
  f for f, keep in zip(feature_names, low_variance_mask_over) if not keep
]
if removed_features_over:
  print(f"  Example removed features (oversampled): {removed_features_over[:5]}")

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 3: Correlation Filter
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 3: Correlation Filter")
print("="*60)

X_train_over_var = X_train_oversampled.loc[:, low_variance_mask_over]
feature_names_var_over = [
  f for f, keep in zip(feature_names, low_variance_mask_over) if keep
]

corr_matrix_over = X_train_over_var.corr().abs()
upper_triangle_over = corr_matrix_over.where(
  np.triu(np.ones(corr_matrix_over.shape), k=1).astype(bool)
)

high_corr_features_over = set()
threshold_over = 0.95

for column in upper_triangle_over.columns:
  correlated = upper_triangle_over.index[upper_triangle_over[column] > threshold_over].tolist()
  if correlated:
    high_corr_features_over.add(column)

print(f"  Correlation threshold (oversampled): {threshold_over}")
print(f"  Highly correlated features to remove (oversampled): {len(high_corr_features_over)}")
print(
  f"  Remaining features (oversampled): "
  f"{len(feature_names_var_over) - len(high_corr_features_over)}"
)

keep_features_corr_over = [
  f for f in feature_names_var_over if f not in high_corr_features_over
]

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 4: Mutual Information
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 4: Mutual Information (Feature Importance)")
print("="*60)

X_train_over_filtered = X_train_oversampled[keep_features_corr_over]

print("  Calculating MI scores on oversampled data (this may take a minute)...")
mi_scores_over = mutual_info_classif(
  X_train_over_filtered, y_train_oversampled, random_state=42
)

mi_df_over = pd.DataFrame(
  {"feature": keep_features_corr_over, "mi_score": mi_scores_over}
).sort_values("mi_score", ascending=False)

print("\n  Top 20 features by Mutual Information (oversampled):")
print(mi_df_over.head(20).to_string(index=False))

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 5: Random Forest Feature Importance
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 5: Random Forest Feature Importance")
print("="*60)

print("  Training Random Forest on oversampled data...")
rf_over = RandomForestClassifier(
  n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
rf_over.fit(X_train_over_filtered, y_train_oversampled)

rf_importance_over = pd.DataFrame(
  {"feature": keep_features_corr_over, "importance": rf_over.feature_importances_}
).sort_values("importance", ascending=False)

print("\n  Top 20 features by Random Forest Importance (oversampled):")
print(rf_importance_over.head(20).to_string(index=False))

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 6: Combined Ranking
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 6: Combined Feature Ranking")
print("="*60)

combined_over = mi_df_over.merge(rf_importance_over, on="feature")
combined_over["mi_rank"] = combined_over["mi_score"].rank(ascending=False)
combined_over["rf_rank"] = combined_over["importance"].rank(ascending=False)
combined_over["avg_rank"] = (
  combined_over["mi_rank"] + combined_over["rf_rank"]
) / 2
combined_over = combined_over.sort_values("avg_rank")

print("\n  Top 30 features (combined ranking, oversampled):")
print(
  combined_over.head(30)[["feature", "mi_score", "importance", "avg_rank"]]
  .to_string(index=False)
)

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 7: Select Top K Features
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 7: Selecting Top Features")
print("="*60)

top_50_over = combined_over.head(50)["feature"].tolist()
top_100_over = combined_over.head(100)["feature"].tolist()
top_200_over = combined_over.head(200)["feature"].tolist()

print("  Top 50 oversampled features selected")
print("  Top 100 oversampled features selected")
print("  Top 200 oversampled features selected")

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 8: PCA Analysis
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 8: PCA Analysis")
print("="*60)

pca_full_over = PCA()
pca_full_over.fit(X_train_over_filtered)

cumulative_variance_over = np.cumsum(pca_full_over.explained_variance_ratio_)

n_90_over = np.argmax(cumulative_variance_over >= 0.90) + 1
n_95_over = np.argmax(cumulative_variance_over >= 0.95) + 1
n_99_over = np.argmax(cumulative_variance_over >= 0.99) + 1

print(f"  Components for 90% variance (oversampled): {n_90_over}")
print(f"  Components for 95% variance (oversampled): {n_95_over}")
print(f"  Components for 99% variance (oversampled): {n_99_over}")

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 9: Visualization
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 9: Creating Visualizations")
print("="*60)

fig_over, axes_over = plt.subplots(2, 2, figsize=(14, 10))
fig_over.suptitle(
  "Feature Selection Analysis (Oversampled)", fontsize=16, fontweight="bold"
)

# 1. Top 20 MI scores (oversampled)
ax1_over = axes_over[0, 0]
top20_mi_over = mi_df_over.head(20)
ax1_over.barh(range(20), top20_mi_over["mi_score"].values, color="steelblue")
ax1_over.set_yticks(range(20))
ax1_over.set_yticklabels(top20_mi_over["feature"].values, fontsize=8)
ax1_over.set_xlabel("Mutual Information Score")
ax1_over.set_title("Top 20 Features (Mutual Information, Oversampled)")
ax1_over.invert_yaxis()

# 2. Top 20 RF importance (oversampled)
ax2_over = axes_over[0, 1]
top20_rf_over = rf_importance_over.head(20)
ax2_over.barh(range(20), top20_rf_over["importance"].values, color="coral")
ax2_over.set_yticks(range(20))
ax2_over.set_yticklabels(top20_rf_over["feature"].values, fontsize=8)
ax2_over.set_xlabel("Feature Importance")
ax2_over.set_title("Top 20 Features (Random Forest, Oversampled)")
ax2_over.invert_yaxis()

# 3. PCA Cumulative Variance (oversampled)
ax3_over = axes_over[1, 0]
ax3_over.plot(
  range(1, len(cumulative_variance_over) + 1),
  cumulative_variance_over,
  "b-",
  linewidth=2,
)
ax3_over.axhline(y=0.90, color="r", linestyle="--", label="90% variance")
ax3_over.axhline(y=0.95, color="g", linestyle="--", label="95% variance")
ax3_over.axvline(x=n_90_over, color="r", linestyle=":", alpha=0.5)
ax3_over.axvline(x=n_95_over, color="g", linestyle=":", alpha=0.5)
ax3_over.set_xlabel("Number of Components")
ax3_over.set_ylabel("Cumulative Explained Variance")
ax3_over.set_title("PCA: Cumulative Variance Explained (Oversampled)")
ax3_over.legend()
ax3_over.set_xlim(0, 200)

# 4. Feature count summary (oversampled)
ax4_over = axes_over[1, 1]
stages_over = [
  "Original",
  "Variance\nFilter",
  "Correlation\nFilter",
  "Top 200",
  "Top 100",
  "Top 50",
]
counts_over = [
  len(feature_names),
  low_variance_mask_over.sum(),
  len(keep_features_corr_over),
  200,
  100,
  50,
]
colors_over = plt.cm.Greens(np.linspace(0.3, 0.9, len(stages_over)))
ax4_over.bar(stages_over, counts_over, color=colors_over)
ax4_over.set_ylabel("Number of Features")
ax4_over.set_title("Feature Reduction Pipeline (Oversampled)")
for i, v in enumerate(counts_over):
  ax4_over.text(i, v + 10, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("feature_selection_analysis_oversampled.png", dpi=150)
plt.show()

print("  Saved: feature_selection_analysis_oversampled.png")

# -----------------------------------------------------------------------------
# OVERSAMPLED STEP 10: Save Selected Features
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("OVERSAMPLED STEP 10: Saving Selected Features")
print("="*60)

os.makedirs("../data/processed/selected", exist_ok=True)

for name, features in [
  ("top50", top_50_over),
  ("top100", top_100_over),
  ("top200", top_200_over),
]:
  X_train_sel_over = X_train_oversampled[features]
  X_val_sel_over = X_val[features]
  X_test_sel_over = X_test[features]

  # Match naming expected by training script for oversampled runs
  X_train_sel_over.to_csv(
    f"../data/processed/selected/X_train_oversampled_{name}.csv",
    index=False,
  )
  X_val_sel_over.to_csv(
    f"../data/processed/selected/X_val_oversampled_{name}.csv",
    index=False,
  )
  X_test_sel_over.to_csv(
    f"../data/processed/selected/X_test_oversampled_{name}.csv",
    index=False,
  )

  pd.DataFrame({"feature": features}).to_csv(
    f"../data/processed/selected/features_oversampled_{name}.csv",
    index=False,
  )

  print(f"  Saved oversampled {name}: {len(features)} features")

combined_over.to_csv(
  "../data/processed/selected/feature_ranking_oversampled.csv", index=False
)
print("  Saved: feature_ranking_oversampled.csv")

print("\n" + "="*60)
print("OVERSAMPLED FEATURE SELECTION COMPLETE - SUMMARY")
print("="*60)
print(f"""
Oversampled Feature Reduction Pipeline:
  1. Original features: {len(feature_names)}
  2. After variance filter: {low_variance_mask_over.sum()} (removed {n_removed_over} low-variance)
  3. After correlation filter: {len(keep_features_corr_over)} (removed {len(high_corr_features_over)} redundant)
  4. Ranking methods: Mutual Information + Random Forest Importance (oversampled)
  5. Selected sets: Top 50, Top 100, Top 200 (oversampled)

Oversampled PCA Analysis:
  - 90% variance: {n_90_over} components
  - 95% variance: {n_95_over} components
  - 99% variance: {n_99_over} components

Top 10 Most Important Features (Oversampled):
{combined_over.head(10)[['feature', 'avg_rank']].to_string(index=False)}

Files Saved (Oversampled):
  - feature_selection_analysis_oversampled.png
  - data/processed/selected/X_train_oversampled_top50.csv (etc.)
  - data/processed/selected/feature_ranking_oversampled.csv

Recommendation:
  - For imbalanced classes, try oversampled Top 100 first
  - Compare performance with non-oversampled Top 100
  - If still overfitting: reduce to oversampled Top 50
""")
