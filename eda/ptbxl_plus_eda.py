import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

sns.set_theme(style="whitegrid", palette="husl")

# =============================================================================
# STEP 1: Load all data
# =============================================================================

# PTB-XL metadata (for labels)
db = pd.read_csv("../ptb-xl/ptbxl_database.csv")
scp = pd.read_csv("../ptb-xl/scp_statements.csv", index_col=0)

# PTB-XL+ features
features_12sl = pd.read_csv("../ptb-xl+/features/12sl_features.csv")
features_ecgdeli = pd.read_csv("../ptb-xl+/features/ecgdeli_features.csv")

print("=== Data Shapes ===")
print(f"PTB-XL metadata: {db.shape}")
print(f"12SL features: {features_12sl.shape}")
print(f"ECGdeli features: {features_ecgdeli.shape}")

# =============================================================================
# STEP 2: Create diagnostic labels
# =============================================================================

db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
scp_diag = scp[scp["diagnostic"] == 1]

def get_diagnostic_superclass(codes_dict):
    for code in codes_dict.keys():
        if code in scp_diag.index:
            return scp_diag.loc[code, "diagnostic_class"]
    return None

db["diagnostic_superclass"] = db["scp_codes"].apply(get_diagnostic_superclass)

# =============================================================================
# STEP 3: Merge features with labels
# =============================================================================

# 12SL features have 'ecg_id' column
df = features_12sl.merge(
    db[["ecg_id", "age", "sex", "diagnostic_superclass", "strat_fold"]], 
    on="ecg_id", 
    how="left"
)

print(f"\n=== Merged Dataset ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.shape[1]} (783 features + 4 metadata)")

# =============================================================================
# STEP 4: Missing value analysis
# =============================================================================

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
missing_df = missing_df[missing_df["count"] > 0].sort_values("count", ascending=False)

print(f"\n=== Missing Values ===")
print(f"Total columns with missing: {len(missing_df)}")
print(f"Top 10 missing:")
print(missing_df.head(10))

# =============================================================================
# STEP 5: Class distribution
# =============================================================================

print(f"\n=== Class Distribution ===")
class_counts = df["diagnostic_superclass"].value_counts()
print(class_counts)
print(f"\nImbalance ratio (max/min): {class_counts.max() / class_counts.min():.2f}x")

# =============================================================================
# STEP 6: Visualizations
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PTB-XL+ Dataset Overview", fontsize=16, fontweight="bold")

# 1. Class distribution
ax1 = axes[0, 0]
colors = sns.color_palette("husl", len(class_counts))
bars = ax1.bar(class_counts.index, class_counts.values, color=colors)
ax1.set_title(f"Diagnostic Superclass Distribution (n={len(df)})")
ax1.set_xlabel("Superclass")
ax1.set_ylabel("Count")
for bar, count in zip(bars, class_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             str(count), ha="center", fontweight="bold")

# 2. Age distribution by class
ax2 = axes[0, 1]
for superclass in class_counts.index:
    subset = df[df["diagnostic_superclass"] == superclass]["age"].dropna()
    sns.kdeplot(subset, ax=ax2, label=superclass, fill=True, alpha=0.3)
ax2.set_title("Age Distribution by Diagnostic Class")
ax2.set_xlabel("Age")
ax2.legend()

# 3. Missing values
ax3 = axes[1, 0]
if len(missing_df) > 0:
    top_missing = missing_df.head(15)
    sns.barplot(x=top_missing["percent"].values, y=top_missing.index, ax=ax3, 
                hue=top_missing.index, legend=False)
    ax3.set_title("Top 15 Columns with Missing Values")
    ax3.set_xlabel("Missing %")
else:
    ax3.text(0.5, 0.5, "No Missing Values!", ha="center", va="center", fontsize=14)
    ax3.set_title("Missing Values")

# 4. Feature correlation sample (first 20 numeric features)
ax4 = axes[1, 1]
numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, ax=ax4, cmap="coolwarm", center=0, 
            xticklabels=False, yticklabels=False)
ax4.set_title("Feature Correlation (First 20 Features)")

plt.tight_layout()
plt.savefig("ptbxl_plus_overview.png", dpi=150)
plt.show()

# =============================================================================
# STEP 7: Feature statistics
# =============================================================================

print(f"\n=== Feature Statistics ===")
numeric_df = df.select_dtypes(include=[np.number])
print(f"Total numeric features: {numeric_df.shape[1]}")

# Key clinical features
clinical_features = [
    "HR__Global",           # Heart rate
    "QRS_Dur_Global",       # QRS duration
    "QT_Int_Global",        # QT interval
    "PR_Int_Global",        # PR interval
    "P_Dur_Global",         # P wave duration
    "P_AxisFront_Global",   # P axis
    "R_AxisFrontal_Global", # QRS axis
    "T_AxisFront_Global"    # T axis
]

print("\nKey Clinical Features:")
for feat in clinical_features:
    if feat in df.columns:
        print(f"  {feat}: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}, missing={df[feat].isnull().sum()}")

# =============================================================================
# STEP 8: Clinical features by class
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Clinical Features by Diagnostic Class", fontsize=16, fontweight="bold")

plot_features = ["HR__Global", "QRS_Dur_Global", "QT_Int_Global", "PR_Int_Global"]
for ax, feat in zip(axes.flat, plot_features):
    if feat in df.columns:
        sns.boxplot(data=df, x="diagnostic_superclass", y=feat, ax=ax, 
                    hue="diagnostic_superclass", legend=False)
        ax.set_title(feat.replace("_Global", "").replace("_", " "))
        ax.set_xlabel("")

plt.tight_layout()
plt.savefig("clinical_features_by_class.png", dpi=150)
plt.show()

print("\n=== EDA Complete ===")
print("Saved: ptbxl_plus_overview.png, clinical_features_by_class.png")
