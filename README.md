# NSL-KDD Anomaly Detection — ML Assignment

## Team & Algorithms
| Member   | Algorithm     | Type        |
|----------|---------------|-------------|
| Member 1 | Random Forest | Supervised  |
| Member 2 | XGBoost       | Supervised  |
| Member 3 | CatBoost      | Supervised  |
| Member 4 | LightGBM      | Supervised  |

## Dataset
File   : `nsl_kdd_dataset.csv`  
Source : https://www.kaggle.com/datasets/programmer3/nsl-kdd-intrusion-detection-dataset  
Shape  : 4430 rows × 42 columns (41 pre-scaled features + label)  
Labels : normal, DoS, Probe, R2L, U2R  
Note   : Features already scaled [0,1] — no further scaling needed.

## Class Imbalance Fix
Raw: 886 normal vs 3544 attack (80% attack ratio).  
Fix: SMOTE oversampling on training set only → balanced 50/50.  
Test set is NEVER oversampled — stays real-world distribution.

## Setup
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost lightgbm shap
```
Place `nsl_kdd_dataset.csv` in the `data/` folder.

## Run Order
1. `shared/00_data_preprocessing.ipynb`          ← ALL members run FIRST
2. `random_forest/03_random_forest.ipynb` ← Member 1
3. `xgboost/04_xgboost.ipynb`             ← Member 2
4. `catboost/01_catboost.ipynb`           ← Member 3
5. `lightgbm/02_lightgbm.ipynb`           ← Member 4
6. `comparison/05_model_comparison.ipynb`         ← ALL members run LAST

## Hyperparameter Tuning (All 4 Models)
- Phase 1: RandomizedSearchCV — 50 iterations, wide parameter space
- Phase 2: GridSearchCV — fine-grained search around best params
- CV: 5-fold Stratified throughout
- Scoring: F1 (optimised for imbalanced intrusion detection)

## Folder Structure
```
ML-assignment/
├── data/
│   ├── nsl_kdd_dataset.csv      ← place your CSV here
│   └── processed/               ← auto-created by notebook 00
├── shared/
│   └── 00_data_preprocessing.ipynb
├── random_forest/
│   └── 03_random_forest.ipynb
├── xgboost/
│   └── 04_xgboost.ipynb
├── catboost/
│   └── 01_catboost.ipynb
├── lightgbm/
│   └── 02_lightgbm.ipynb
├── comparison/
│   └── 05_model_comparison.ipynb
├── report/
│   └── ML_Assignment_Report.pdf
├── members.txt
├── submission.txt
└── README.md
```