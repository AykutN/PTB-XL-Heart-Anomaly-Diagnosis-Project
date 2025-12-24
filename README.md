# PTB-XL+ ECG Diagnosis Recognition

Multi-class ECG classification using machine learning on the PTB-XL+ dataset, focusing on handling missing data, feature selection, and class imbalance.

## Project Structure

```
PTB-XL+ paper/
├── preprocessing/          # Data preprocessing scripts
│   ├── pipeline.py        # Complete preprocessing pipeline
│   ├── feature_selection.py  # Feature selection (Top-50, Top-100, Top-200)
│   ├── merge_and_label.py
│   ├── clean_columns.py
│   ├── split_data.py
│   └── impute_and_flag.py
│
├── training/              # Model training scripts
│   └── train_models.py    # Model training and evaluation
│
├── figures/               # Figure generation
│   └── generate_figures.py
│
├── data/
│   └── processed/        # Preprocessed data (train/val/test splits)
│
├── reports/
│   ├── feature_selection/    # Feature selection plots and lists
│   ├── missing_data_analysis/ # Missing data visualizations
│   ├── model_comparison/     # Model performance plots
│   └── outlier_analysis/     # Outlier analysis plots
│
├── results/              # CSV results of model experiments
│
├── src/                  # Additional analysis scripts
│   ├── analysis/         # Analysis scripts
│   ├── modeling/         # Additional modeling scripts
│   └── preprocessing/    # Original preprocessing scripts
│
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets:**
   - PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
   - PTB-XL+: https://physionet.org/content/ptb-xl-plus/1.0.1/
   
   Place the datasets in the project root:
   - `ptb-xl/` - PTB-XL dataset
   - `ptb-xl+/` - PTB-XL+ feature dataset

   **Note:** Processed CSV files are not included in the repository. You must run the preprocessing pipeline to generate them.

## Pipeline & Usage

The project follows a sequential pipeline. Run scripts in order:

### 1. Preprocessing

Run the complete preprocessing pipeline:

```bash
python preprocessing/pipeline.py
```

This script performs:
- Merging PTB-XL metadata with PTB-XL+ features
- Processing labels and creating multi-label columns (NORM, MI, STTC, CD, HYP)
- Cleaning columns (dropping sparse/metadata columns)
- Splitting data into train/validation/test sets using stratified folds
- Imputing missing values and creating missing flags

### 2. Feature Selection

Select top features using Random Forest importance:

```bash
python preprocessing/feature_selection.py
```

This generates:
- Top-50, Top-100, and Top-200 feature lists
- Feature importance plots
- Feature selection reports

### 3. Model Training

Train and evaluate models:

```bash
python training/train_models.py
```

This script:
- Trains Random Forest and SVM models with class-weight approach
- Uses Top-200 features
- Performs threshold optimization
- Generates performance metrics and visualizations

### 4. Generate Figures

Generate all figures for the paper:

```bash
python figures/generate_figures.py
```

## Methodology

### Missing Data Handling
- **Analysis:** Identified that missingness in P-wave features is correlated with MI and STTC classes (MNAR).
- **Strategy:** Missing values were imputed with 0 (for signal absence) or median, and binary flags (`is_missing_*`) were added to preserve the information of "missingness".

### Feature Selection
- **VarianceThreshold:** Removed features with variance < 0.05.
- **Random Forest Importance:** Selected the top 200 features from the remaining set.

### Class Imbalance Handling
- **Class Weighting:** Assigning higher weights to minority classes (HYP, CD) using `class_weight='balanced'`.
- **Threshold Optimization:** Optimizing classification thresholds per class to maximize F1 score.

### Models Evaluated
1. **Random Forest** with class weights
2. **SVM** with class weights

## Results

The best performance was achieved using **SVM** with **Top-200 features** and **Threshold Optimization**.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 58.85% | 71.30% | 75.60% | 73.21% | 0.915 |
| **SVM** | **59.08%** | **71.06%** | **78.99%** | **74.61%** | **0.922** |

*Note: Metrics are macro-averaged.*

## Reproducibility

All experiments can be reproduced by running the scripts in order:
1. `preprocessing/pipeline.py`
2. `preprocessing/feature_selection.py`
3. `training/train_models.py`
4. `figures/generate_figures.py`

Results are saved in:
- `results/` - CSV files with performance metrics
- `reports/` - Visualizations and analysis reports

## Code Environment

- **Python:** 3.11+
- **Key Libraries:**
  - pandas, numpy - Data processing
  - scikit-learn - Modeling and evaluation
  - matplotlib, seaborn - Visualization

## References

1. Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data.
2. Strodthoff, N., et al. (2023). PTB-XL+, a comprehensive electrocardiographic feature dataset. Scientific Data.

## License

This project is for educational purposes at Istanbul University.
