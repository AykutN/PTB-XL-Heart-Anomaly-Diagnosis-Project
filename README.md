# PTB-XL ECG Classification Project

Multi-class ECG classification using machine learning on the PTB-XL dataset, focusing on handling missing data, feature selection, and class imbalance.

## Project Structure

```
Machine Learning IU/
├── Article/
│   └── paper.md                 # Scientific report (Turkish)
│
├── data/
│   └── processed/               # Preprocessed data (train/val/test splits)
│
├── reports/
│   ├── feature_selection/       # Feature selection plots and lists
│   ├── missing_data_analysis/   # Missing data visualizations
│   └── model_comparison/        # Model performance plots (ROC, Precision/Recall)
│
├── results/                     # CSV results of model experiments
│
├── src/
│   ├── analysis/                # Analysis scripts
│   │   ├── missing_data_analysis.py
│   │   ├── feature_selection_rf.py
│   │   └── ...
│   ├── modeling/                # Modeling scripts
│   │   ├── 06_train_models.py
│   │   ├── 08_threshold_optimization.py
│   │   ├── 11_advanced_experiments.py
│   │   └── 12_balanced_rf.py
│   └── preprocessing/           # Data cleaning and splitting scripts
│
├── ptb-xl/                      # Raw PTB-XL dataset
├── ptb-xl+/                     # PTB-XL+ feature dataset
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets:**
   - PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
   - PTB-XL+: https://physionet.org/content/ptb-xl-plus/1.0.1/

## Pipeline & Usage

The project follows a sequential pipeline. Scripts are located in `src/`.

### 1. Preprocessing
- **Merge & Label:** Merges PTB-XL metadata with PTB-XL+ features and assigns diagnostic superclasses.
- **Clean & Split:** Drops irrelevant columns, handles missing data (flagging + imputation), and splits into Train/Val/Test.

### 2. Feature Selection
- **Variance Threshold:** Removes low-variance features (< 0.05).
- **Random Forest Importance:** Selects Top-50, Top-100, and Top-200 features based on importance scores.

### 3. Modeling
- **Standard Random Forest:** Baseline model with class weights.
- **Ensemble Undersampling:** 50-seed ensemble with undersampling for majority classes.
- **Balanced Random Forest:** Uses `imblearn` for balanced bootstrapping.
- **Threshold Optimization:** Optimizes classification thresholds per class to maximize F1 score.

## Methodology

### Missing Data Handling
- **Analysis:** Identified that missingness in P-wave features is correlated with MI and STTC classes (MNAR).
- **Strategy:** Instead of dropping rows, missing values were imputed with 0 (for signal absence) or median, and binary flags (`is_missing_*`) were added to preserve the information of "missingness".

### Feature Selection
- **VarianceThreshold:** Removed 186 features with variance < 0.05.
- **Random Forest Importance:** Selected the top 200 features from the remaining set. PCA was avoided to maintain interpretability.

### Class Imbalance Handling
- **Class Weighting:** Assigning higher weights to minority classes (HYP, CD).
- **Ensemble Undersampling:** Training multiple models on balanced subsets to reduce bias towards the majority class (NORM).
- **Balanced Random Forest:** Automatically balances bootstrap samples.

### Models Evaluated
1. **Random Forest (Baseline)**
2. **Random Forest + Class Weights**
3. **Random Forest + Ensemble Undersampling**
4. **Balanced Random Forest**

## Results

The best performance was achieved using **Balanced Random Forest** with **Top-200 features** and **Threshold Optimization**.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard RF | 58.85% | 71.30% | 75.60% | 73.21% | 0.915 |
| **Balanced RF** | **60.94%** | **73.66%** | **77.08%** | **75.29%** | **0.925** |

*Note: Metrics are macro-averaged.*

## Report

The detailed scientific report (in Turkish) is available at `Article/paper.md`.

## References

1. Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data.
2. Strodthoff, N., et al. (2021). PTB-XL+, a comprehensive electrocardiographic feature dataset. Scientific Data.

## License

This project is for educational purposes at Istanbul University.
