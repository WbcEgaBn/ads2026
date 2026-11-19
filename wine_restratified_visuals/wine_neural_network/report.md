# Wine quality neural network: training and validation

Source: `wine2.csv` (SHA-256: `ff8898e24f19f48a77ffe8f5a7a0d2784388f184f4e4a4bf3276841eeebaacff`).

Target: numerical quality score. Inputs: 11 chemical measurements plus red/white wine type.
Removed 0 identical rows from 5,320 source rows.
Development rows: 4,788; reserved test rows: 532.
**No test-set prediction or evaluation is performed by this workflow.** A new class-stratified holdout is reserved.
Source rows overlap earlier experiments; this is not a historically unseen test set.

Class-stratified 5-fold cross-validation uses only development data. Identical inputs stay together.
Each fold uses an inner class-stratified approximately 80/20 split to select an epoch count by early stopping on validation MAE.
A fresh network then trains on the full outer training fold for that epoch count before outer validation scoring.
Scalers and target means are fitted separately using only the corresponding training rows. quality_class is excluded.
Every inner fit, outer refit, and final fit uses each original training row once per epoch, with no class resampling.
KNN and regression baselines also use original training rows without duplication or class weighting.
Scaling and target centering use only original training rows. Splits preserve observed class proportions.
Training metrics describe the original training rows; validation folds retain their original class proportions.
Final unique training class counts: {'Low': 213, 'Medium': 3667, 'High': 908}.
Selected architecture: [32, 16] (lowest mean fold MAE).
Final model trained on all development rows for 58 epochs (rounded median of selected fold epochs).

| Model | Validation MAE, mean ± SD | RMSE, mean ± SD | Exact accuracy, mean ± SD | Within one point, mean ± SD |
|---|---:|---:|---:|---:|
| Neural network 32-16 | 0.542 ± 0.016 | 0.696 ± 0.022 | 56.0% ± 1.8% | 96.1% ± 0.7% |
| Neural network 64-32 | 0.546 ± 0.011 | 0.702 ± 0.015 | 54.9% ± 2.0% | 96.0% ± 0.5% |
| Mean baseline | 0.695 ± 0.009 | 0.880 ± 0.014 | 43.7% ± 1.1% | 92.7% ± 0.6% |
| Ridge regression | 0.566 ± 0.013 | 0.733 ± 0.018 | 53.8% ± 1.5% | 94.9% ± 0.9% |

Selected model pooled out-of-fold MAE: 0.542; RMSE: 0.696; exact accuracy: 56.0%; within one point: 96.1%.
Final model in-sample training MAE: 0.513; exact accuracy: 57.9%. This is a different fit from the fold models.

MAE and RMSE use continuous predictions. Accuracy uses half-up rounding clipped to the 0–10 quality scale.
Fold SD describes variation across folds, not a confidence interval. Pooled metrics weight rows; fold means weight folds equally.
Cross-validation also selects the architecture, so selected-model scores may be optimistic and are not independent test estimates.
Rare quality scores are underrepresented. See validation_metrics_by_quality.csv and validation_metrics_by_wine_type.csv.

Reproduce: `python3 preprocessing_restratified.py --model neural-network --folds 5 --epochs 400 --patience 40`

Plots: learning_curves.png, split_performance.png, validation_predictions.png, baseline_comparison.png, cross_validation.png, training_class_distribution.png.
Saved final model: model.pt; preprocessing: scaler.joblib. predictions.csv contains selected-model out-of-fold predictions only.
cv_predictions.csv contains all candidates and baselines. fold_metrics.csv and fold_assignments.csv provide the validation audit trail.
final_history.csv describes the final full-development fit; history_*_fold_*.csv files describe inner early stopping.
training_rows.csv records unique class counts and confirms zero repeated source indices for every model fit.
partition_assignments.csv and split_class_distribution.csv audit outer, inner, and reserved partitions.
This separate workflow is implemented in preprocessing_restratified.py. Legacy files and outputs are unchanged.
