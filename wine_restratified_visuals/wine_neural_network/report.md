# Wine quality neural network classifier

Source: `wine2.csv` (SHA-256: `ff8898e24f19f48a77ffe8f5a7a0d2784388f184f4e4a4bf3276841eeebaacff`).

Target: quality_class — Low, Medium, or High. Numerical quality is unused and is not required in the input CSV.
Inputs: 11 chemical measurements plus red/white wine type. Both quality and quality_class are excluded from features.
The network outputs three logits, trains with cross-entropy, and uses softmax probabilities in Low/Medium/High order.
The predicted class has the highest probability; no score rounding or quality-score thresholds are used.

Removed 0 duplicate feature/label rows from 5,320 source rows.
Development rows: 4,788; reserved test rows: 532.
This training workflow does not evaluate reserved tests. Run testing.py for saved-model test results.
Source rows overlap earlier experiments.
Class-stratified 5-fold validation preserves identical-input groups.
Each outer fold uses an inner approximately 80/20 stratified split to choose epochs by maximum validation macro F1.
A fresh network refits the outer training rows before outer validation scoring. Scaling uses training rows only.
Each training row is used once per epoch; no resampling or class weighting is applied, including for baselines.
Final training class counts: {'Low': 213, 'Medium': 3667, 'High': 908}.
Selected architecture: [64, 32] (highest mean outer-fold macro F1).
Final development fit: 73 epochs (rounded median of selected fold epochs).

| Model | Accuracy, mean ± SD | Macro F1, mean ± SD | Macro ROC AUC, mean ± SD |
|---|---:|---:|---:|
| Neural network 32-16 | 79.2% ± 0.9% | 0.4881 ± 0.0195 | 0.8018 ± 0.0127 |
| Neural network 64-32 | 79.4% ± 0.5% | 0.4958 ± 0.0378 | 0.8094 ± 0.0183 |
| Majority baseline | 76.6% ± 0.0% | 0.2891 ± 0.0001 | 0.5000 ± 0.0000 |
| Logistic regression | 78.2% ± 1.1% | 0.4430 ± 0.0168 | 0.7913 ± 0.0182 |

Selected model pooled out-of-fold accuracy: **79.4%**; macro F1: **0.4980**; macro ROC AUC: **0.8079**.
Final model in-sample accuracy: 80.7%. This is a different fit from the fold models.

Macro F1 gives each class equal weight. ROC curves use out-of-fold probabilities, with one class versus the rest.
Fold SD describes variation across folds, not a confidence interval. Pooled metrics weight rows; fold means weight folds equally.
Validation selects the architecture, so selected-model scores may be optimistic and are not independent test estimates.
KNN predicts the same target but uses different partitions; its scores are not a paired model comparison.

Reproduce: `python3 preprocessing_restratified.py --model neural-network --folds 5 --epochs 400 --patience 40`

predictions.csv contains selected-model out-of-fold class predictions and three probabilities; cv_predictions.csv includes all models.
validation_metrics_by_class.csv contains precision, recall, F1, and support. confusion_matrix.csv contains class counts.
ROC outputs: roc_curves.png/.csv, roc_auc_by_class.csv, roc_auc_by_fold.png; metrics.json records pooled and fold summaries.
Plots: learning_curves.png, split_performance.png, validation_predictions.png, baseline_comparison.png, cross_validation.png, training_class_distribution.png.
model.pt stores the final classifier and its class order; scaler.joblib stores its fitted preprocessing.
fold_metrics.csv, fold_assignments.csv, history_*_fold_*.csv, and final_history.csv audit validation and training.
training_rows.csv, partition_assignments.csv, and split_class_distribution.csv audit training counts and partition membership.
