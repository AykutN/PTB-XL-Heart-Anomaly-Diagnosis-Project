import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
INPUT_CSV = os.path.join(BASE_DIR, 'reports/missing_data_analysis/missingness_diagnosis_correlation.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports/missing_data_analysis')

def main():
    # Load data
    df = pd.read_csv(INPUT_CSV)
    
    # Filter out insignificant features (very low missing count)
    # T_AxisFront_Global (4) and R_AxisFrontal_Global (1) are negligible
    # Also drop 'heart_axis' as requested
    df_filtered = df[(df['Missing_Count'] > 100) & (df['Feature'] != 'heart_axis')].copy()
    
    # Prepare data for heatmap
    # We want rows=Features, cols=Diagnostics, values=Percentage
    
    heatmap_data = pd.DataFrame(index=df_filtered['Feature'])
    
    diag_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    for diag in diag_cols:
        col_name = f'{diag}_in_Missing(%)'
        heatmap_data[diag] = df_filtered[col_name].values
        
    # Clean up feature names (remove _Global suffix for cleaner look, but keep English)
    heatmap_data.index = [x.replace('_Global', '') for x in heatmap_data.index]
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.2)
    
    # Create heatmap
    # Annotate with percentage values
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Bulunma Oranı (%)'})
    
    plt.title('Eksik Veri İçeren Kayıtların Tanı Sınıflarına Göre Dağılımı', pad=20)
    plt.ylabel('Eksik Öznitelik')
    plt.xlabel('Tanı Sınıfı')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'missingness_diagnosis_heatmap_paper.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved heatmap to {output_path}")

if __name__ == "__main__":
    main()
