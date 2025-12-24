import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FEATURE_DIR = os.path.join(BASE_DIR, 'reports/feature_selection/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports/outlier_analysis/')

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_imputed.csv'), index_col='ecg_id')
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    return train_df, target_cols

def load_features(n):
    path = os.path.join(FEATURE_DIR, f'top{n}_features.csv')
    df = pd.read_csv(path)
    return df['Feature'].tolist()

def main():
    train_df, target_cols = load_data()
    features = load_features(200) # Use Top-200 features for analysis
    
    X = train_df[features]
    y = train_df[target_cols]
    
    # Standardize data before Isolation Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Isolation Forest
    print("Running Isolation Forest...")
    # contamination='auto' usually sets it around 0.1, or we can specify.
    # Let's use 'auto' to let the algorithm decide based on score distribution.
    iso = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    outlier_preds = iso.fit_predict(X_scaled)
    
    # -1 is outlier, 1 is inlier
    train_df['is_outlier'] = outlier_preds
    train_df['outlier_label'] = train_df['is_outlier'].map({1: 'Normal', -1: 'Aykırı (Outlier)'})
    
    num_outliers = (outlier_preds == -1).sum()
    total_samples = len(X)
    print(f"Total Samples: {total_samples}")
    print(f"Detected Outliers: {num_outliers} ({num_outliers/total_samples*100:.2f}%)")
    
    # 2. Analyze Outliers per Diagnostic Class
    print("Analyzing outliers per class...")
    
    # Since it's multi-label, we iterate over classes
    class_outlier_stats = []
    
    for col in target_cols:
        # Get samples belonging to this class
        class_samples = train_df[train_df[col] == 1]
        n_class = len(class_samples)
        n_outliers_class = (class_samples['is_outlier'] == -1).sum()
        ratio = n_outliers_class / n_class * 100
        
        class_outlier_stats.append({
            'Class': col,
            'Total': n_class,
            'Outliers': n_outliers_class,
            'Ratio (%)': ratio
        })
        
    stats_df = pd.DataFrame(class_outlier_stats)
    print("\nOutlier Statistics per Class:")
    print(stats_df)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'outlier_stats_per_class.csv'), index=False)
    
    # 3. Visualizations
    
    # Bar Plot: Outlier Ratio per Class
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Ratio (%)', data=stats_df, palette='Reds')
    plt.title('Tanı Sınıflarına Göre Aykırı Değer Oranı')
    plt.ylabel('Aykırı Değer Oranı (%)')
    plt.xlabel('Tanı Sınıfı')
    plt.ylim(0, 100)
    for index, row in stats_df.iterrows():
        plt.text(index, row['Ratio (%)'] + 1, f"{row['Ratio (%)']:.1f}%", ha='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_ratio_per_class.png'), dpi=300)
    print(f"Saved bar plot to {os.path.join(OUTPUT_DIR, 'outlier_ratio_per_class.png')}")
    
    # PCA Visualization (2D)
    print("Generating PCA visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Outlier'] = train_df['outlier_label'].values
    
    # We also want to color by "Main Class" (simplified for viz)
    # Priority: MI > STTC > CD > HYP > NORM
    def get_main_label(row):
        if row['MI'] == 1: return 'MI'
        if row['STTC'] == 1: return 'STTC'
        if row['CD'] == 1: return 'CD'
        if row['HYP'] == 1: return 'HYP'
        if row['NORM'] == 1: return 'NORM'
        return 'Other'
    
    pca_df['Diagnosis'] = train_df.apply(get_main_label, axis=1)
    
    plt.figure(figsize=(12, 8))
    # Color by Diagnosis, Style by Outlier status
    sns.scatterplot(x='PC1', y='PC2', hue='Diagnosis', style='Outlier', data=pca_df, 
                    palette={'NORM': 'green', 'MI': 'red', 'STTC': 'orange', 'CD': 'purple', 'HYP': 'blue', 'Other': 'gray'},
                    alpha=0.7, s=60)
    plt.title('PCA Düzleminde Veri Dağılımı ve Aykırı Değerler')
    plt.xlabel(f'PC1 (Var: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 (Var: {pca.explained_variance_ratio_[1]:.2f})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_outliers_by_class.png'), dpi=300)
    print(f"Saved PCA plot to {os.path.join(OUTPUT_DIR, 'pca_outliers_by_class.png')}")
    
    # Boxplot of top 3 features for Outliers vs Normals
    top_3_features = features[:3]
    
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(top_3_features):
        plt.subplot(1, 3, i+1)
        sns.boxplot(x='outlier_label', y=feature, data=train_df, palette={'Normal': 'skyblue', 'Aykırı (Outlier)': 'salmon'})
        plt.title(feature)
        plt.xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_features_boxplot.png'), dpi=300)
    print(f"Saved boxplot to {os.path.join(OUTPUT_DIR, 'outlier_features_boxplot.png')}")

if __name__ == "__main__":
    main()
