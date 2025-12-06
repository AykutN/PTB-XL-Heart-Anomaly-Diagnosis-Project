# PTB-XL ECG Classification Project

Multi-class ECG classification using machine learning on the PTB-XL dataset.

## Project Structure

```
Machine Learning IU/
├── Article/
│   └── paper.md                 # Scientific report (Turkish)
│
├── data/
│   └── processed/               # Preprocessed data
│       ├── X_train.csv         # Training features
│       ├── X_val.csv           # Validation features
│       ├── X_test.csv          # Test features
│       ├── y_train.csv         # Training labels
│       ├── y_val.csv           # Validation labels
│       ├── y_test.csv          # Test labels
│       ├── class_weights.csv   # Computed class weights
│       ├── feature_names.csv   # Feature names
│       └── selected/           # Selected feature sets
│           ├── X_train_top50.csv
│           ├── X_train_top100.csv
│           ├── X_train_top200.csv
│           └── ...
│
├── eda/
│   ├── metadata.py             # Basic EDA on PTB-XL metadata
│   ├── ptbxl_plus_eda.py       # EDA on PTB-XL+ features
│   ├── ptbxl_plus_overview.png
│   ├── clinical_features_by_class.png
│   ├── distributions.png
│   └── top_diagnoses.png
│
├── figures/
│   ├── generate_figures.py     # Figure generation script
│   ├── preprocessing_pipeline.png
│   ├── class_distribution_splits.png
│   ├── missing_values_analysis.png
│   ├── class_weights.png
│   └── feature_selection_summary.png
│
├── preprocessing/
│   ├── pipeline.py             # Main preprocessing pipeline
│   ├── feature_selection.py    # Feature selection & extraction
│   └── feature_selection_analysis.png
│
├── training/
│   └── train_models.py         # Model training script
│
├── models/                     # Saved trained models (after training)
├── results/                    # Evaluation results (after training)
│
├── ptb-xl/                     # Raw PTB-XL dataset
├── ptb-xl+/                    # PTB-XL+ feature dataset
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets:**
   - PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
   - PTB-XL+: https://physionet.org/content/ptb-xl-plus/1.0.1/

## Usage

### 1. Preprocessing

Run the preprocessing pipeline to clean data and extract features:

```bash
python preprocessing/pipeline.py
```

This will:
- Load PTB-XL metadata and PTB-XL+ features
- Extract diagnostic labels
- Handle missing values (hybrid approach)
- Split data into train/val/test
- Apply StandardScaler normalization
- Save processed data

### 2. Feature Selection

Run feature selection to reduce dimensionality:

```bash
python preprocessing/feature_selection.py
```

This will:
- Remove highly correlated features
- Compute mutual information and random forest importance
- Create top 50/100/200 feature sets
- Perform PCA analysis
- Generate visualization

### 3. Generate Figures

Create figures for the report:

```bash
python figures/generate_figures.py
```

### 4. Train Models

Train and evaluate classifiers:

```bash
python training/train_models.py
```

This will:
- Train Decision Tree, Naive Bayes, and SVM
- Perform hyperparameter tuning with grid search
- Evaluate on test set
- Save models and results

## Dataset

### PTB-XL

- 21,799 clinical 12-lead ECG recordings
- 18,869 unique patients
- 10-second recordings at 100/500 Hz
- Labeled with SCP-ECG codes

### Diagnostic Classes

| Class | Description | Count | Percentage |
|-------|-------------|-------|------------|
| NORM | Normal ECG | 9,514 | 44.5% |
| MI | Myocardial Infarction | 5,424 | 25.4% |
| STTC | ST/T Change | 2,817 | 13.2% |
| CD | Conduction Disturbance | 2,325 | 10.9% |
| HYP | Hypertrophy | 1,308 | 6.1% |

## Methods

### Preprocessing
- Missing value handling: Hybrid approach (median imputation + missing indicators)
- Feature scaling: StandardScaler (fit on training data only)
- Class imbalance: Weighted learning

### Feature Selection
- Correlation filtering (>0.95)
- Mutual Information
- Random Forest importance
- Combined ranking

### Classification Algorithms
1. **Decision Tree**: Interpretable, feature importance
2. **Gaussian Naive Bayes**: Fast, probabilistic
3. **SVM**: Effective in high dimensions

### Evaluation Metrics
- Accuracy
- Macro/Weighted F1-Score
- Per-class Precision/Recall
- Confusion Matrix
- ROC-AUC

## Results

*To be completed after model training*

## Report

The scientific report (in Turkish) is available at `Article/paper.md`.

## References

1. Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data.
2. Strodthoff, N., et al. (2021). PTB-XL+, a comprehensive electrocardiographic feature dataset. Scientific Data.

## License

This project is for educational purposes at Istanbul University.
