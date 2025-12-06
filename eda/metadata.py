import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

sns.set_theme(style="whitegrid", palette="husl")

ptbxl_database = "../ptb-xl/ptbxl_database.csv"
scp_statements = "../ptb-xl/scp_statements.csv"

db = pd.read_csv(ptbxl_database)
scp = pd.read_csv(scp_statements, index_col=0)

# Parse scp_codes from string to dictionary
db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

# Get diagnostic classes
scp_diag = scp[scp["diagnostic"] == 1]

def get_diagnostic_superclass(codes_dict):
    for code in codes_dict.keys():
        if code in scp_diag.index:
            return scp_diag.loc[code, "diagnostic_class"]
    return None

db["diagnostic_superclass"] = db["scp_codes"].apply(get_diagnostic_superclass)

# --- Figure 1: Dataset Overview ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f"PTB-XL Dataset Overview (n={len(db)})", fontsize=16, fontweight="bold")

# Age distribution
sns.histplot(db["age"].dropna(), bins=30, kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title(f"Age Distribution\nMean: {db['age'].mean():.1f}, Std: {db['age'].std():.1f}")
axes[0, 0].set_xlabel("Age (years)")

# Sex distribution
sex_counts = db["sex"].value_counts()
sex_labels = ["Male" if x == 0 else "Female" for x in sex_counts.index]
sns.barplot(x=sex_labels, y=sex_counts.values, ax=axes[0, 1], hue=sex_labels, legend=False)
axes[0, 1].set_title(f"Sex Distribution\nMale: {sex_counts.get(0, 0)}, Female: {sex_counts.get(1, 0)}")
for i, v in enumerate(sex_counts.values):
    axes[0, 1].text(i, v + 100, str(v), ha="center", fontweight="bold")

# Height distribution
sns.histplot(db["height"].dropna(), bins=30, kde=True, ax=axes[0, 2], color="coral")
axes[0, 2].set_title(f"Height Distribution\nMean: {db['height'].mean():.1f} cm")
axes[0, 2].set_xlabel("Height (cm)")

# Weight distribution
sns.histplot(db["weight"].dropna(), bins=30, kde=True, ax=axes[1, 0], color="mediumseagreen")
axes[1, 0].set_title(f"Weight Distribution\nMean: {db['weight'].mean():.1f} kg")
axes[1, 0].set_xlabel("Weight (kg)")

# Diagnostic superclass distribution
superclass_counts = db["diagnostic_superclass"].value_counts()
sns.barplot(x=superclass_counts.index, y=superclass_counts.values, ax=axes[1, 1], hue=superclass_counts.index, legend=False)
axes[1, 1].set_title("Diagnostic Superclass Distribution")
axes[1, 1].set_xlabel("Superclass")
for i, v in enumerate(superclass_counts.values):
    axes[1, 1].text(i, v + 100, str(v), ha="center", fontweight="bold")

# Missing values
missing = db.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    sns.barplot(x=missing.values, y=missing.index, ax=axes[1, 2], hue=missing.index, legend=False)
    axes[1, 2].set_title("Missing Values")
    axes[1, 2].set_xlabel("Count")
else:
    axes[1, 2].text(0.5, 0.5, "No Missing Values", ha="center", va="center", fontsize=14)
    axes[1, 2].set_title("Missing Values")

plt.tight_layout()
plt.savefig("dataset_overview.png", dpi=150)
plt.show()

# --- Figure 2: Top Diagnoses ---
from collections import Counter

all_codes = []
for codes in db["scp_codes"]:
    all_codes.extend(codes.keys())

code_counts = Counter(all_codes)
top_10 = code_counts.most_common(10)

fig, ax = plt.subplots(figsize=(12, 6))
codes, counts = zip(*top_10)
sns.barplot(x=list(codes), y=list(counts), ax=ax, hue=list(codes), legend=False)
ax.set_title("Top 10 Diagnosis Codes", fontsize=14, fontweight="bold")
ax.set_xlabel("SCP Code")
ax.set_ylabel("Count")
for i, v in enumerate(counts):
    ax.text(i, v + 100, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("top_diagnoses.png", dpi=150)
plt.show()