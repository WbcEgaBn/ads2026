# Wine quality neural network: training and validation

Source: `wine2.csv` (SHA-256: `ff8898e24f19f48a77ffe8f5a7a0d2784388f184f4e4a4bf3276841eeebaacff`).

Target: numerical quality score. Inputs: 11 chemical measurements plus red/white wine type.
Removed 0 identical rows from 5,320 source rows.
Development rows: 4,788; reserved test rows: 532.
**No test-set prediction or evaluation is performed by this workflow.** The original test assignment is preserved.
Earlier repository outputs evaluated this partition; it is not a historically unseen test set.

Grouped 5-fold cross-validation uses only development data. Identical inputs stay together.
Each fold uses an inner grouped 80/20 split to select an epoch count by early stopping on inner validation MAE.
A fresh network then trains on the full outer training fold for that epoch count before outer validation scoring.
Scalers and target means are fitted separately using only the corresponding training rows. quality_class is excluded.
For every inner fit, outer refit, and final fit, Low and High rows are sampled with replacement to match the Medium training count.
The same balancing helper is used by KNN and the neural network. Baselines also use balanced training rows.
Scaling fits the original training rows before resampling; target centering uses the balanced training targets.
Training metrics describe the original training rows; validation folds retain their original class proportions.
Final training counts after resampling: {'Low': 3665, 'Medium': 3665, 'High': 3665}.
Selected architecture: [64, 32] (lowest mean fold MAE).
Final model trained on all development rows for 5 epochs (rounded median of selected fold epochs).

| Model | Validation MAE, mean ± SD | RMSE, mean ± SD | Exact accuracy, mean ± SD | Within one point, mean ± SD |
|---|---:|---:|---:|---:|
| Neural network 32-16 | 0.641 ± 0.017 | 0.820 ± 0.026 | 48.0% ± 1.2% | 93.5% ± 0.4% |
| Neural network 64-32 | 0.638 ± 0.019 | 0.816 ± 0.024 | 47.5% ± 1.5% | 93.5% ± 0.8% |
| Mean baseline | 0.762 ± 0.030 | 0.920 ± 0.041 | 43.7% ± 1.7% | 92.5% ± 1.4% |
| Ridge regression | 0.645 ± 0.019 | 0.829 ± 0.034 | 48.1% ± 1.6% | 92.8% ± 1.1% |

Selected model pooled out-of-fold MAE: 0.638; RMSE: 0.817; exact accuracy: 47.5%; within one point: 93.5%.
Final model in-sample training MAE: 0.603; exact accuracy: 50.0%. This is a different fit from the fold models.

MAE and RMSE use continuous predictions. Accuracy uses half-up rounding clipped to the 0–10 quality scale.
Fold SD describes variation across folds, not a confidence interval. Pooled metrics weight rows; fold means weight folds equally.
Cross-validation also selects the architecture, so selected-model scores may be optimistic and are not independent test estimates.
Rare quality scores are underrepresented. See validation_metrics_by_quality.csv and validation_metrics_by_wine_type.csv.

Reproduce: `python3 train_wine_neural_network.py --folds 5 --epochs 400 --patience 40`

Plots: learning_curves.png, split_performance.png, validation_predictions.png, baseline_comparison.png, cross_validation.png, training_class_balance.png.
Saved final model: model.pt; preprocessing: scaler.joblib. predictions.csv contains selected-model out-of-fold predictions only.
cv_predictions.csv contains all candidates and baselines. fold_metrics.csv and fold_assignments.csv provide the validation audit trail.
final_history.csv describes the final full-development fit; history_*_fold_*.csv files describe inner early stopping.
training_balance.csv records class counts before and after resampling for every model fit.
Training implementations and the shared resampling helper are in preprocessing.py.
