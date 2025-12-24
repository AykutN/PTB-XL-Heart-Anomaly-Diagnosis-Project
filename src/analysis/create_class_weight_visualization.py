"""
Create visualization showing class weights assigned to each class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
REPORT_DIR = os.path.join(BASE_DIR, 'reports/model_comparison/')

os.makedirs(REPORT_DIR, exist_ok=True)

def calculate_class_weights(train_df, target_cols):
    """Calculate class weights using sklearn's balanced formula."""
    class_counts = train_df[target_cols].sum()
    total_samples = len(train_df)
    n_classes = len(target_cols)
    
    weights = {}
    for col in target_cols:
        count = class_counts[col]
        # sklearn formula: n_samples / (n_classes * np.bincount(y))
        weight = total_samples / (n_classes * count)
        weights[col] = weight
    
    return class_counts, weights

def create_class_weight_visualization():
    """Create visualization of class weights."""
    print("Loading training data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_imputed.csv'), index_col='ecg_id')
    target_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Calculate class counts and weights
    class_counts, weights = calculate_class_weights(train_df, target_cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Class counts bar plot
    ax1 = axes[0]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    counts_array = class_counts.values
    bars = ax1.bar(target_cols, counts_array, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts_array):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({count/len(train_df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Örnek Sayısı', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tanı Sınıfı', fontsize=12, fontweight='bold')
    ax1.set_title('Sınıf Dağılımı (Training Set)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(counts_array) * 1.15])
    
    # 2. Class weights bar plot
    ax2 = axes[1]
    weight_values = [weights[col] for col in target_cols]
    bars2 = ax2.bar(target_cols, weight_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, weight in zip(bars2, weight_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add horizontal line at weight=1.0 (baseline)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (1.0)')
    
    ax2.set_ylabel('Class Weight', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Tanı Sınıfı', fontsize=12, fontweight='bold')
    ax2.set_title('Class-Weight Değerleri (Balanced)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(weight_values) * 1.15])
    
    plt.suptitle('Class-Weight Yaklaşımı: Sınıf Dağılımı ve Ağırlıklar', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(REPORT_DIR, 'class_weights_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved class-weight visualization to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLASS-WEIGHT SUMMARY")
    print("="*60)
    print(f"Total training samples: {len(train_df)}")
    print(f"Number of classes: {len(target_cols)}")
    print("\nClass Details:")
    for col in target_cols:
        count = class_counts[col]
        weight = weights[col]
        percentage = count / len(train_df) * 100
        print(f"  {col:4s}: Count={count:5d} ({percentage:5.2f}%), Weight={weight:.4f}")
    
    plt.close()

if __name__ == "__main__":
    create_class_weight_visualization()

