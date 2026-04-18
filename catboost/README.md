# CatBoost (Member 3)

NSL-KDD anomaly detection using CatBoost gradient boosting.

## Prerequisites

```bash
pip install catboost shap imbalanced-learn pandas numpy matplotlib seaborn scikit-learn scipy
```

## Run order

1. **First**, run `shared/00_data_preprocessing.ipynb` (produces `data/processed/preprocessed_data.pkl`).
2. Then run this notebook: `catboost/01_catboost.ipynb`.
3. Finally, the team can run `comparison/05_model_comparison.ipynb` to aggregate all four models.

## Expected runtime

- Phase 1 RandomizedSearchCV: ~5-10 min on a 4-core i5 laptop (20 candidates x 5 folds = 100 fits).
- Phase 2 GridSearchCV: ~3-6 min.
- Final training + CV + plots + SHAP: ~2-3 min.
- **Total: roughly 10-20 minutes on CPU.**

## Outputs

All artefacts are written to the shared `comparison/` folder so the team comparison notebook can pick them up:

| File                                            | Purpose                                       |
|-------------------------------------------------|-----------------------------------------------|
| `comparison/results_catboost.csv`               | Binary metrics in the team-comparable schema  |
| `comparison/results_catboost_macro.csv`         | Macro-averaged metrics (honest report)        |
| `comparison/cv_scores_catboost.png`             | Per-fold CV F1, accuracy, ROC-AUC             |
| `comparison/cm_roc_catboost.png`                | Confusion matrix + ROC curve                  |
| `comparison/feature_importance_catboost.png`    | Top-15 native CatBoost importances            |
| `comparison/shap_catboost.png`                  | SHAP summary on a 500-sample test subset      |
| `comparison/learning_curve_catboost.png`        | F1 per iteration (train vs. test)             |

## Files in this folder

- `01_catboost.ipynb` - the notebook (single deliverable).
- `REPORT_SECTION.md` - report-ready writeup (paste into the team report).
- `README.md` - this file.
- `catboost_info/` - CatBoost's own training logs (auto-generated, can be ignored).
