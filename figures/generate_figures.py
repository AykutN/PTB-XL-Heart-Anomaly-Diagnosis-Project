"""
Generate Figures for Paper

This script generates all figures used in the paper:
- Class distribution visualizations
- Feature selection plots
- Model comparison plots
- Outlier analysis plots

Usage:
    python figures/generate_figures.py
"""

import sys
import os

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'analysis'))

from create_class_weight_visualization import create_class_weight_visualization
from outlier_analysis_per_label import main as outlier_analysis

def main():
    """Generate all figures for the paper."""
    print("="*80)
    print("GENERATING FIGURES FOR PAPER")
    print("="*80)
    
    # 1. Class weight visualization
    print("\n[1/2] Generating class weight distribution figure...")
    create_class_weight_visualization()
    
    # 2. Outlier analysis figures
    print("\n[2/2] Generating outlier analysis figures...")
    outlier_analysis()
    
    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETE")
    print("="*80)
    print("\nFigures saved to:")
    print("  - reports/model_comparison/class_weights_distribution.png")
    print("  - reports/outlier_analysis/")

if __name__ == "__main__":
    main()

