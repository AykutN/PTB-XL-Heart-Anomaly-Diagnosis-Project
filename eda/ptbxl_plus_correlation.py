import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


sns.set_theme(style="whitegrid")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "ptb-xl+" / "features" / "12sl_features.csv"
    fig_dir = base_dir / "figures"
    output_dir = base_dir / "eda"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

    corr = numeric_df.corr()
    abs_corr = corr.abs()
    upper_mask = np.triu(np.ones_like(abs_corr, dtype=bool), k=1)
    upper = abs_corr.where(upper_mask)

    top_pairs = (
        upper.stack()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_corr"})
    )
    top_pairs_path = output_dir / "ptbxl_plus_top_corr_pairs.csv"
    top_pairs.head(50).to_csv(top_pairs_path, index=False)

    patterns = ["_Global", "Axis", "HR_", "RR_", "PR_", "QT_", "QRS_", "P_Dur", "T_Dur"]
    subset_cols = [col for col in numeric_df.columns if any(pat in col for pat in patterns)]
    if len(subset_cols) < 2:
        subset_cols = numeric_df.columns.tolist()

    subset_df = numeric_df[subset_cols]
    subset_abs_corr = subset_df.corr().abs()
    feature_scores = subset_abs_corr.sum().sort_values(ascending=False)
    heatmap_cols = feature_scores.head(min(30, len(feature_scores))).index
    heatmap_corr = subset_df[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        heatmap_corr,
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_title("PTB-XL+ 12SL Global Özellik Korelasyonu")
    plt.tight_layout()
    heatmap_path = fig_dir / "ptbxl_plus_global_corr_heatmap.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    pairs_to_plot = top_pairs.head(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (_, row) in zip(axes.flatten(), pairs_to_plot.iterrows()):
        feat_a, feat_b, value = row
        pair_df = numeric_df[[feat_a, feat_b]].dropna()
        sample_size = min(5000, len(pair_df))
        if sample_size == 0:
            ax.text(0.5, 0.5, "No data", ha="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        pair_df = pair_df.sample(n=sample_size, random_state=42)
        sns.scatterplot(data=pair_df, x=feat_a, y=feat_b, ax=ax, s=10, alpha=0.4)
        ax.set_title(f"{feat_a} vs {feat_b}\n|r|={value:.2f}")
    plt.tight_layout()
    scatter_path = fig_dir / "ptbxl_plus_top_corr_pairs.png"
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    print(f"Saved heatmap to: {heatmap_path}")
    print(f"Saved scatter plots to: {scatter_path}")
    print(f"Saved top correlation table to: {top_pairs_path}")


if __name__ == "__main__":
    main()
