# KNN training and validation

Source: `wine2.csv` (SHA-256: `ff8898e24f19f48a77ffe8f5a7a0d2784388f184f4e4a4bf3276841eeebaacff`).

Uses the original raw rows and stratified 80/20 development/test split. No test evaluation is performed.

StandardScaler is fitted separately on each training fold. Every original training row is used once per fit, without class resampling. The neural network also uses original training rows. Stratified partitions preserve observed class proportions.

Fixed K: 3 (specified by the user); mean 10-fold macro F1: 0.5139 ± 0.0319. No search over K is performed. These are validation scores, not reserved-test results.

## ROC AUC

One-vs-rest ROC AUC uses held-out fold probabilities, with explicit Low/Medium/High column alignment. Every development row is scored once by a model trained on the other folds. The reserved test rows are excluded.

Pooled out-of-fold macro AUC: **0.7236**; weighted AUC: 0.7384; micro AUC: 0.8935.

Mean fold macro AUC: 0.7240 ± 0.0273 SD. The pooled score combines validation rows; the fold mean gives each fold equal weight.

| Class vs rest | Pooled ROC AUC |
|---|---:|
| Low | 0.6531 |
| Medium | 0.7315 |
| High | 0.7862 |

Macro averages the three class AUCs equally; weighted averages use observed class counts; micro pools all row/class pairs. AUC measures ranking across thresholds, not classification accuracy. K is fixed at 3. K is not selected using these validation folds.

Charts: roc_curves.png, roc_auc_by_fold.png, macro_f1_by_fold.png. Numeric outputs: roc_auc_by_class.csv, roc_curves.csv, fold_metrics.csv and metrics.json. predictions.csv includes the fixed K=3 model's out-of-fold class probabilities.

See training_rows.csv and split_class_distribution.csv for exact class counts and macro_f1_by_fold.png for validation performance. KNN and neural network predict the same classes but use different partitions; these are not paired comparison scores.
