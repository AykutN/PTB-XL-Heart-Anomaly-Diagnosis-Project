"""
Comprehensive Outlier Analysis for Each Target Label
Generates publication-quality visualizations for outlier detection per diagnostic class.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FEATURE_DIR = os.path.join(BASE_DIR, 'reports/feature_selection/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports/outlier_analysis/')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load training data and target labels."""
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_imputed.csv'), index_col='ecg_id')
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    return train_df, target_cols

def load_features(n):
    """Load top N features from feature selection results."""
    path = os.path.join(FEATURE_DIR, f'top{n}_features.csv')
    df = pd.read_csv(path)
    return df['Feature'].tolist()

def detect_outliers_isolation_forest(X_scaled, contamination=0.1):
    """Detect outliers using Isolation Forest."""
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    outlier_preds = iso.fit_predict(X_scaled)
    return outlier_preds == -1  # True for outliers

def detect_outliers_iqr(X, threshold=1.5):
    """Detect outliers using IQR method for each feature."""
    outlier_mask = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[1]):
        Q1 = np.percentile(X[:, i], 25)
        Q3 = np.percentile(X[:, i], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        feature_outliers = (X[:, i] < lower_bound) | (X[:, i] > upper_bound)
        outlier_mask |= feature_outliers
    return outlier_mask

def detect_outliers_zscore(X_scaled, threshold=3):
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(X_scaled, axis=0))
    outlier_mask = (z_scores > threshold).any(axis=1)
    return outlier_mask

def analyze_outliers_per_label(train_df, features, target_cols):
    """Analyze outliers for each target label separately."""
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS PER TARGET LABEL")
    print("="*80)
    
    X = train_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    all_results = {}
    
    for label in target_cols:
        print(f"\n--- Analyzing outliers for {label} ---")
        
        # Get samples for this label
        label_mask = train_df[label] == 1
        label_indices = train_df[label_mask].index
        X_label = X[label_mask]
        X_label_scaled = X_scaled[label_mask]
        
        n_samples = len(X_label)
        print(f"  Total samples: {n_samples}")
        
        if n_samples < 10:
            print(f"  Skipping {label}: insufficient samples")
            continue
        
        # Detect outliers using multiple methods
        # Isolation Forest (10% contamination)
        iso_outliers = detect_outliers_isolation_forest(X_label_scaled, contamination=0.1)
        n_iso = iso_outliers.sum()
        
        # IQR method
        iqr_outliers = detect_outliers_iqr(X_label, threshold=1.5)
        n_iqr = iqr_outliers.sum()
        
        # Z-score method
        zscore_outliers = detect_outliers_zscore(X_label_scaled, threshold=3)
        n_zscore = zscore_outliers.sum()
        
        # Consensus: outlier if detected by at least 2 methods
        consensus_outliers = (iso_outliers.astype(int) + 
                             iqr_outliers.astype(int) + 
                             zscore_outliers.astype(int)) >= 2
        n_consensus = consensus_outliers.sum()
        
        print(f"  Isolation Forest outliers: {n_iso} ({n_iso/n_samples*100:.2f}%)")
        print(f"  IQR outliers: {n_iqr} ({n_iqr/n_samples*100:.2f}%)")
        print(f"  Z-score outliers: {n_zscore} ({n_zscore/n_samples*100:.2f}%)")
        print(f"  Consensus outliers (≥2 methods): {n_consensus} ({n_consensus/n_samples*100:.2f}%)")
        
        # Store results
        all_results[label] = {
            'indices': label_indices,
            'X': X_label,
            'X_scaled': X_label_scaled,
            'iso_outliers': iso_outliers,
            'iqr_outliers': iqr_outliers,
            'zscore_outliers': zscore_outliers,
            'consensus_outliers': consensus_outliers,
            'n_samples': n_samples,
            'n_iso': n_iso,
            'n_iqr': n_iqr,
            'n_zscore': n_zscore,
            'n_consensus': n_consensus
        }
    
    return all_results, scaler

def create_visualizations_per_label(all_results, features, target_cols, scaler):
    """Create publication-quality visualizations for each label."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Color palette for labels
    label_colors = {
        'NORM': '#2ecc71',  # Green
        'MI': '#e74c3c',    # Red
        'STTC': '#f39c12',  # Orange
        'CD': '#9b59b6',    # Purple
        'HYP': '#3498db'    # Blue
    }
    
    # 1. Summary statistics table
    summary_data = []
    for label in target_cols:
        if label not in all_results:
            continue
        res = all_results[label]
        summary_data.append({
            'Label': label,
            'Total Samples': res['n_samples'],
            'Isolation Forest (%)': f"{res['n_iso']/res['n_samples']*100:.2f}",
            'IQR (%)': f"{res['n_iqr']/res['n_samples']*100:.2f}",
            'Z-score (%)': f"{res['n_zscore']/res['n_samples']*100:.2f}",
            'Consensus (%)': f"{res['n_consensus']/res['n_samples']*100:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'outlier_summary_per_label.csv'), index=False)
    print(f"\nSaved summary statistics to {os.path.join(OUTPUT_DIR, 'outlier_summary_per_label.csv')}")
    
    # 2. Bar plot: Outlier percentages per label
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(target_cols))
    width = 0.2
    
    methods = ['Isolation Forest', 'IQR', 'Z-score', 'Consensus']
    colors_methods = ['#3498db', '#e67e22', '#1abc9c', '#c0392b']
    
    for i, method in enumerate(methods):
        values = []
        for label in target_cols:
            if label in all_results:
                res = all_results[label]
                if method == 'Isolation Forest':
                    val = res['n_iso'] / res['n_samples'] * 100
                elif method == 'IQR':
                    val = res['n_iqr'] / res['n_samples'] * 100
                elif method == 'Z-score':
                    val = res['n_zscore'] / res['n_samples'] * 100
                else:  # Consensus
                    val = res['n_consensus'] / res['n_samples'] * 100
                values.append(val)
            else:
                values.append(0)
        
        ax.bar(x + i*width, values, width, label=method, color=colors_methods[i], alpha=0.8)
    
    ax.set_xlabel('Diagnostic Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outlier Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Outlier Detection by Method Across Diagnostic Classes', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(target_cols)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_comparison_all_labels.png'), dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {os.path.join(OUTPUT_DIR, 'outlier_comparison_all_labels.png')}")
    plt.close()
    
    # 3. Individual visualizations for each label
    for label in target_cols:
        if label not in all_results:
            continue
        
        print(f"\nCreating visualizations for {label}...")
        res = all_results[label]
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 3a. PCA visualization with outliers highlighted
        ax1 = fig.add_subplot(gs[0, :])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(res['X_scaled'])
        
        # Plot inliers and outliers separately
        inlier_mask = ~res['consensus_outliers']
        outlier_mask = res['consensus_outliers']
        
        ax1.scatter(X_pca[inlier_mask, 0], X_pca[inlier_mask, 1], 
                   c=label_colors[label], alpha=0.5, s=30, label='Inliers', edgecolors='none')
        ax1.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], 
                   c='red', alpha=0.8, s=80, marker='x', linewidths=2, label='Outliers')
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=11, fontweight='bold')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{label}: PCA Visualization of Outliers', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # 3b. Box plots of top 5 features
        ax2 = fig.add_subplot(gs[1, 0])
        top_features = features[:5]
        feature_data = []
        feature_names_short = []
        outlier_labels = []
        
        for feat in top_features:
            feat_idx = features.index(feat)
            feature_data.extend([res['X'][inlier_mask, feat_idx], res['X'][outlier_mask, feat_idx]])
            feature_names_short.append(feat[:20] + '...' if len(feat) > 20 else feat)
            outlier_labels.extend(['Inlier', 'Outlier'])
        
        # Create box plot data
        plot_data = []
        plot_labels = []
        for i, feat in enumerate(top_features):
            feat_idx = features.index(feat)
            plot_data.append(res['X'][inlier_mask, feat_idx])
            plot_data.append(res['X'][outlier_mask, feat_idx])
            plot_labels.append(f'{feat[:15]}...\nInlier' if len(feat) > 15 else f'{feat}\nInlier')
            plot_labels.append(f'{feat[:15]}...\nOutlier' if len(feat) > 15 else f'{feat}\nOutlier')
        
        bp = ax2.boxplot(plot_data, labels=plot_labels, patch_artist=True, 
                         widths=0.6, showfliers=False)
        
        # Color boxes
        for i, patch in enumerate(bp['boxes']):
            if i % 2 == 0:  # Inlier
                patch.set_facecolor(label_colors[label])
                patch.set_alpha(0.6)
            else:  # Outlier
                patch.set_facecolor('red')
                patch.set_alpha(0.6)
        
        ax2.set_ylabel('Feature Value', fontsize=10, fontweight='bold')
        ax2.set_title(f'{label}: Top 5 Features (Inliers vs Outliers)', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3c. Outlier detection method comparison
        ax3 = fig.add_subplot(gs[1, 1])
        methods = ['Isolation\nForest', 'IQR', 'Z-score', 'Consensus\n(≥2)']
        counts = [res['n_iso'], res['n_iqr'], res['n_zscore'], res['n_consensus']]
        percentages = [c/res['n_samples']*100 for c in counts]
        
        bars = ax3.bar(methods, percentages, color=['#3498db', '#e67e22', '#1abc9c', '#c0392b'], alpha=0.8)
        ax3.set_ylabel('Outlier Percentage (%)', fontsize=10, fontweight='bold')
        ax3.set_title(f'{label}: Outlier Detection Methods Comparison', fontsize=11, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%\n(n={int(height*res["n_samples"]/100)})',
                    ha='center', va='bottom', fontsize=9)
        
        # 3d. Distribution of outlier scores (using Isolation Forest scores)
        ax4 = fig.add_subplot(gs[2, 0])
        iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        iso.fit(res['X_scaled'])
        scores = iso.score_samples(res['X_scaled'])
        
        ax4.hist(scores[inlier_mask], bins=30, alpha=0.6, color=label_colors[label], 
                label='Inliers', edgecolor='black', linewidth=0.5)
        ax4.hist(scores[outlier_mask], bins=30, alpha=0.8, color='red', 
                label='Outliers', edgecolor='black', linewidth=0.5)
        ax4.axvline(np.median(scores), color='black', linestyle='--', linewidth=2, label='Median')
        ax4.set_xlabel('Isolation Forest Score', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax4.set_title(f'{label}: Outlier Score Distribution', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        # 3e. Summary statistics text
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        stats_text = f"""
        OUTLIER ANALYSIS SUMMARY: {label}
        
        Total Samples: {res['n_samples']:,}
        
        Detection Methods:
        • Isolation Forest: {res['n_iso']} ({res['n_iso']/res['n_samples']*100:.2f}%)
        • IQR Method: {res['n_iqr']} ({res['n_iqr']/res['n_samples']*100:.2f}%)
        • Z-score Method: {res['n_zscore']} ({res['n_zscore']/res['n_samples']*100:.2f}%)
        • Consensus (≥2 methods): {res['n_consensus']} ({res['n_consensus']/res['n_samples']*100:.2f}%)
        
        Feature Space:
        • Dimensions: {res['X'].shape[1]}
        • PCA Explained Variance (2D): {pca.explained_variance_ratio_.sum():.2%}
        """
        
        ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Comprehensive Outlier Analysis: {label} Class', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, f'outlier_analysis_{label}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {output_path}")
        plt.close()
    
    print("\nAll visualizations generated successfully!")

def main():
    """Main function to run outlier analysis."""
    print("="*80)
    print("OUTLIER ANALYSIS PER TARGET LABEL")
    print("="*80)
    
    # Load data
    train_df, target_cols = load_data()
    features = load_features(200)  # Use Top-200 features
    
    print(f"\nLoaded {len(train_df)} samples with {len(features)} features")
    print(f"Target labels: {', '.join(target_cols)}")
    
    # Analyze outliers per label
    all_results, scaler = analyze_outliers_per_label(train_df, features, target_cols)
    
    # Create visualizations
    create_visualizations_per_label(all_results, features, target_cols, scaler)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

