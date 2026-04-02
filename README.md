# NSL-KDD Anomaly Detection — ML Assignment

## Dataset
File: `nsl_kdd_dataset.csv`  
Source: https://www.kaggle.com/datasets/programmer3/nsl-kdd-intrusion-detection-dataset  
Shape: 4430 rows × 42 columns (41 pre-scaled features + label column)  
Labels: normal, DoS, Probe, R2L, U2R  
Note: Features are already scaled to [0, 1] — no further scaling needed.

## Setup
Place `nsl_kdd_dataset.csv` in the `data/` folder.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost lightgbm shap
```

## Run Order

1. `shared/00_data_preprocessing.ipynb`           ← ALL members run this first
2. `catboost/01_catboost.ipynb`   ← Member 1
3. `lightgmb/02_lightgmb.ipynb`                       ← Member 2
4. `random_forest/03_random_forest.ipynb`         ← Member 3
5. `xgboost/04_xgboost.ipynb`                     ← Member 4
6. `comparison/05_model_comparison.ipynb`          ← ALL members run this last

## What Each Notebook Does

| Notebook | Key steps |
|---|---|
| 00_preprocessing | Load CSV, EDA, binary labels, 80/20 split, save .pkl |
| 01_isolation_forest | Unsupervised anomaly detection, contamination tuning, ROC, PCA viz |
| 02_dbscan | PCA reduction, k-dist elbow, clustering, eps sensitivity |
| 03_random_forest | Supervised RF, cross-val, feature importance, ROC |
| 04_xgboost | GridSearch tuning, SHAP, learning curve, ROC |
| 05_comparison | Load all results, comparison table, bar chart, heatmap |

## Folder Structure

```
ML-assignment/
├── data/
│   ├── nsl_kdd_dataset.csv      ← place your CSV here
│   └── processed/               ← auto-created by notebook 00
├── shared/
│   └── 00_data_preprocessing.ipynb
├── member1_isolation_forest/
│   └── 01_isolation_forest.ipynb
├── member2_dbscan/
│   └── 02_dbscan.ipynb
├── member3_random_forest/
│   └── 03_random_forest.ipynb
├── member4_xgboost/
│   └── 04_xgboost.ipynb
├── comparison/
│   └── 05_model_comparison.ipynb
├── report/
│   └── ML_Assignment_Report.pdf
├── members.txt
├── submission.txt
└── README.md
```
