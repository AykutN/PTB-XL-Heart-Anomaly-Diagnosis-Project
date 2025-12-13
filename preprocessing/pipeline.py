import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import ast
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1: Load Data
# =============================================================================

print("="*60)
print("STEP 1: Loading Data")
print("="*60)

db = pd.read_csv("../ptb-xl/ptbxl_database.csv")
scp = pd.read_csv("../ptb-xl/scp_statements.csv", index_col=0)
features = pd.read_csv("../ptb-xl+/features/12sl_features.csv")

print(f"  Metadata: {db.shape}")
print(f"  Features: {features.shape}")

# =============================================================================
# STEP 2: Create Labels
# =============================================================================

print("\n" + "="*60)
print("STEP 2: Creating Labels")
print("="*60)

db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
scp_diag = scp[scp["diagnostic"] == 1]

def get_diagnostic_superclass(codes_dict):
    for code in codes_dict.keys():
        if code in scp_diag.index:
            return scp_diag.loc[code, "diagnostic_class"]
    return None

db["label"] = db["scp_codes"].apply(get_diagnostic_superclass)

# Merge features with labels and metadata
df = features.merge(
    db[["ecg_id", "label", "strat_fold", "age", "sex", "height", "weight"]], 
    on="ecg_id", 
    how="left"
)

print(f"  Merged shape: {df.shape}")

# =============================================================================
# STEP 3: Remove rows with missing labels
# =============================================================================

print("\n" + "="*60)
print("STEP 3: Removing Rows with Missing Labels")
print("="*60)

before = len(df)
df = df.dropna(subset=["label"])
#! here we droped the NaN values. why? -ib
after = len(df)

print(f"  Removed: {before - after} rows ({(before-after)/before*100:.2f}%)")
print(f"  Remaining: {after} rows")

# =============================================================================
# STEP 4: Row-wise missing pattern analysis
# =============================================================================

print("\n" + "="*60)
print("STEP 4: Row-wise Missing Pattern Analysis")
print("="*60)

# Calculate missing ratio per row
exclude_cols = ["ecg_id", "label", "strat_fold", "age", "sex", "height", "weight"]
feature_cols_temp = [col for col in df.columns if col not in exclude_cols]

row_missing = df[feature_cols_temp].isnull().sum(axis=1)
row_missing_ratio = row_missing / len(feature_cols_temp)

print(f"  Max missing ratio per row: {row_missing_ratio.max():.2%}")
print(f"  Rows with >50% missing: {(row_missing_ratio > 0.5).sum()}")
print(f"  Rows with >25% missing: {(row_missing_ratio > 0.25).sum()}")
print(f"  Rows with >10% missing: {(row_missing_ratio > 0.1).sum()}")

# Remove rows with >50% missing features (if any)
high_missing_mask = row_missing_ratio > 0.5
if high_missing_mask.sum() > 0:
    print(f"  Removing {high_missing_mask.sum()} rows with >50% missing features")
    df = df[~high_missing_mask]
else:
    print("  No rows with >50% missing features - keeping all")

# =============================================================================
# STEP 5: Create BMI feature from height/weight
# =============================================================================

print("\n" + "="*60)
print("STEP 5: Feature Engineering - BMI")
print("="*60)

# BMI = weight (kg) / (height (cm) / 100)^2
df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)

print(f"  BMI created: mean={df['BMI'].mean():.2f}, missing={df['BMI'].isnull().sum()}")
print(f"  Height missing: {df['height'].isnull().sum()}")
print(f"  Weight missing: {df['weight'].isnull().sum()}")

# =============================================================================
# STEP 6: Create Missing Indicator Features (Hybrid Approach)
# =============================================================================

print("\n" + "="*60)
print("STEP 6: Creating Missing Indicator Features")
print("="*60)

# These features have clinically meaningful missingness
# P wave missing = likely AF, flutter, junctional rhythm
clinical_missing_features = [
    "P_On_Global",
    "P_Off_Global", 
    "P_Dur_Global",
    "PR_Int_Global",
    "P_AxisFront_Global",
    "HR_Atrial_Global"
]

# Create binary missing indicators
for feat in clinical_missing_features:
    if feat in df.columns:
        missing_col = f"{feat}_missing"
        df[missing_col] = df[feat].isnull().astype(int)
        print(f"  Created {missing_col}: {df[missing_col].sum()} missing ({df[missing_col].mean()*100:.2f}%)")

# =============================================================================
# STEP 7: Define feature groups
# =============================================================================

print("\n" + "="*60)
print("STEP 7: Defining Feature Groups")
print("="*60)

# Columns to exclude from features
meta_cols = ["ecg_id", "label", "strat_fold"]

# Clinical features from PTB-XL+ (will be imputed)
feature_cols = [col for col in df.columns 
                if col not in meta_cols 
                and not col.endswith("_missing")]

# Missing indicator features (no imputation needed)
missing_indicator_cols = [col for col in df.columns if col.endswith("_missing")]

print(f"  Numeric features: {len(feature_cols)}")
print(f"  Missing indicators: {len(missing_indicator_cols)}")
print(f"  Total features: {len(feature_cols) + len(missing_indicator_cols)}")

# =============================================================================
# STEP 8: Train/Val/Test Split (BEFORE imputation - critical!)
# =============================================================================

print("\n" + "="*60)
print("STEP 8: Train/Val/Test Split (Before Imputation)")
print("="*60)

# PTB-XL predefined folds: 1-8 = train, 9 = val, 10 = test
train_mask = df["strat_fold"] <= 8
val_mask = df["strat_fold"] == 9
test_mask = df["strat_fold"] == 10

df_train = df[train_mask].copy()
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()

print(f"  Train: {len(df_train)} samples ({len(df_train)/len(df)*100:.1f}%)")
print(f"  Val:   {len(df_val)} samples ({len(df_val)/len(df)*100:.1f}%)")
print(f"  Test:  {len(df_test)} samples ({len(df_test)/len(df)*100:.1f}%)")

# Separate X and y
X_train = df_train[feature_cols]
X_val = df_val[feature_cols]
X_test = df_test[feature_cols]

y_train = df_train["label"]
y_val = df_val["label"]
y_test = df_test["label"]

# Missing indicators (no imputation needed)
X_train_missing = df_train[missing_indicator_cols]
X_val_missing = df_val[missing_indicator_cols]
X_test_missing = df_test[missing_indicator_cols]

# =============================================================================
# STEP 9: Imputation - FIT ONLY ON TRAIN SET (No Data Leakage!)
# =============================================================================

print("\n" + "="*60)
print("STEP 9: Imputation (Train-only fit - No Data Leakage)")
print("="*60)

# Check missing before
print(f"  Missing values before imputation:")
print(f"    Train: {X_train.isnull().sum().sum()}")
print(f"    Val:   {X_val.isnull().sum().sum()}")
print(f"    Test:  {X_test.isnull().sum().sum()}")

# Fit imputer ONLY on train set
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)  # FIT only on train!

# Transform all sets
X_train_imputed = pd.DataFrame(
    imputer.transform(X_train),
    columns=feature_cols,
    index=X_train.index
)
X_val_imputed = pd.DataFrame(
    imputer.transform(X_val),
    columns=feature_cols,
    index=X_val.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_cols,
    index=X_test.index
)

print(f"\n  Missing values after imputation:")
print(f"    Train: {X_train_imputed.isnull().sum().sum()}")
print(f"    Val:   {X_val_imputed.isnull().sum().sum()}")
print(f"    Test:  {X_test_imputed.isnull().sum().sum()}")

# =============================================================================
# STEP 10: Feature Scaling - FIT ONLY ON TRAIN SET
# =============================================================================

print("\n" + "="*60)
print("STEP 10: Feature Scaling (Train-only fit)")
print("="*60)

scaler = StandardScaler()
scaler.fit(X_train_imputed)  # FIT only on train!

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train_imputed),
    columns=feature_cols,
    index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val_imputed),
    columns=feature_cols,
    index=X_val.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imputed),
    columns=feature_cols,
    index=X_test.index
)

print(f"  Train mean: {X_train_scaled.mean().mean():.6f} (should be ~0)")
print(f"  Train std:  {X_train_scaled.std().mean():.6f} (should be ~1)")
print(f"  Val mean:   {X_val_scaled.mean().mean():.6f} (slightly off is OK)")
print(f"  Test mean:  {X_test_scaled.mean().mean():.6f} (slightly off is OK)")

# =============================================================================
# STEP 11: Combine scaled features with missing indicators
# =============================================================================

print("\n" + "="*60)
print("STEP 11: Combining Features with Missing Indicators")
print("="*60)

X_train_final = pd.concat([X_train_scaled.reset_index(drop=True), X_train_missing.reset_index(drop=True)], axis=1)
X_val_final = pd.concat([X_val_scaled.reset_index(drop=True), X_val_missing.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_scaled.reset_index(drop=True), X_test_missing.reset_index(drop=True)], axis=1)

print(f"  Final feature count: {X_train_final.shape[1]}")
print(f"    - Numeric features: {len(feature_cols)}")
print(f"    - Missing indicators: {len(missing_indicator_cols)}")

# =============================================================================
# STEP 12: Class Distribution & Weights
# =============================================================================

print("\n" + "="*60)
print("STEP 12: Class Distribution & Weights")
print("="*60)

from sklearn.utils.class_weight import compute_class_weight

print("  Class distribution in each split:")
for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    counts = y.value_counts()
    print(f"\n  {name}:")
    for cls in sorted(counts.index):
        print(f"    {cls}: {counts[cls]} ({counts[cls]/len(y)*100:.1f}%)")

# Compute class weights from train set only
classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

print("\n  Class weights (computed from train):")
for cls, weight in sorted(class_weight_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"    {cls}: {weight:.3f}")

# =============================================================================
# STEP 13: Save preprocessed data
# =============================================================================

print("\n" + "="*60)
print("STEP 13: Saving Preprocessed Data")
print("="*60)

import os
os.makedirs("../data/processed", exist_ok=True)

# Save as CSV
X_train_final.to_csv("../data/processed/X_train.csv", index=False)
X_val_final.to_csv("../data/processed/X_val.csv", index=False)
X_test_final.to_csv("../data/processed/X_test.csv", index=False)

y_train.to_csv("../data/processed/y_train.csv", index=False)
y_val.to_csv("../data/processed/y_val.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)

# Save class weights
pd.DataFrame([class_weight_dict]).to_csv("../data/processed/class_weights.csv", index=False)

# Save feature names
all_features = list(feature_cols) + list(missing_indicator_cols)
pd.DataFrame({"feature": all_features}).to_csv("../data/processed/feature_names.csv", index=False)

print("  Files saved to ../data/processed/")

# =============================================================================
# STEP 14: Summary
# =============================================================================

print("\n" + "="*60)
print("PREPROCESSING COMPLETE - SUMMARY")
print("="*60)
print(f"""
Dataset:
  - Total samples: {len(df)}
  - Features: {len(all_features)} ({len(feature_cols)} numeric + {len(missing_indicator_cols)} missing indicators)
  - Classes: {len(classes)} ({', '.join(sorted(classes))})

Splits:
  - Train: {len(X_train_final)} ({len(X_train_final)/len(df)*100:.1f}%)
  - Val:   {len(X_val_final)} ({len(X_val_final)/len(df)*100:.1f}%)
  - Test:  {len(X_test_final)} ({len(X_test_final)/len(df)*100:.1f}%)

Preprocessing Steps:
  1. Removed rows with missing labels (411 rows)
  2. Row-wise missing check (removed rows with >50% missing)
  3. Created BMI feature from height/weight
  4. Created missing indicator features for P-wave related columns
  5. Train/Val/Test split BEFORE imputation (no data leakage)
  6. Median imputation (fitted on train only)
  7. StandardScaler (fitted on train only)
  8. Computed class weights for imbalance handling

Missing Indicators Created:
  {', '.join(missing_indicator_cols)}

Class Weights:
  {', '.join([f'{k}: {v:.2f}' for k, v in sorted(class_weight_dict.items(), key=lambda x: x[1], reverse=True)])}
""")
