import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define paths
BASE_DIR = '/Users/y.aykut/Codebase/PTB-XL+ paper/'
INPUT_CSV = os.path.join(BASE_DIR, 'results/per_class_metrics_comparison.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports/model_comparison')

def main():
    print("Loading metrics...")
    df = pd.read_csv(INPUT_CSV, index_col=0) # Index is class name
    
    # Reset index to make 'Class' a column
    df = df.reset_index().rename(columns={'index': 'Class'})
    
    # Melt dataframe for seaborn plotting
    # We want to compare Precision and Recall for both models
    # Let's create two separate plots or one combined?
    # Combined is better: X=Class, Y=Score, Hue=Model, Style=Metric?
    # Or separate plots for Precision and Recall.
    
    # Let's do separate plots for clarity
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    titles = {
        'precision': 'Kesinlik (Precision) Karşılaştırması',
        'recall': 'Duyarlılık (Recall) Karşılaştırması',
        'f1-score': 'F1 Skoru Karşılaştırması'
    }
    
    sns.set_context("paper", font_scale=1.2)
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        sns.barplot(x='Class', y=metric, hue='Model', data=df, palette='viridis')
        
        plt.title(titles[metric])
        plt.ylabel('Skor')
        plt.xlabel('Tanı Sınıfı')
        plt.ylim(0, 1.0)
        plt.legend(title='Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, f'comparison_{metric}.png')
        plt.savefig(output_path, dpi=300)
        print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
