# KNN training and validation

Source: `wine2.csv` (SHA-256: `ff8898e24f19f48a77ffe8f5a7a0d2784388f184f4e4a4bf3276841eeebaacff`).

Uses the original raw rows and stratified 80/20 development/test split. No test evaluation is performed.

StandardScaler is fitted separately on each training fold. Medium training rows are retained; Low and High are resampled with replacement to match Medium. The neural network uses the same balancing helper in preprocessing.py. Validation rows retain their original distribution.

Selected K: 3; mean 10-fold macro F1: 0.5324 ± 0.0415. This score also selects K and is not an independent test estimate.

## ROC AUC

One-vs-rest ROC AUC uses held-out fold probabilities, with explicit Low/Medium/High column alignment. Every development row is scored once by a model trained on the other folds. The reserved test rows are excluded.

Pooled out-of-fold macro AUC: **0.7157**; weighted AUC: 0.7284; micro AUC: 0.8365.

Mean fold macro AUC: 0.7157 ± 0.0293 SD. The pooled score combines validation rows; the fold mean gives each fold equal weight.

| Class vs rest | Pooled ROC AUC |
|---|---:|
| Low | 0.6487 |
| Medium | 0.7208 |
| High | 0.7778 |

Macro averages the three class AUCs equally; weighted averages use observed class counts; micro pools all row/class pairs. AUC measures ranking across thresholds, not classification accuracy. K remains selected by macro F1, not AUC. These selected-model validation scores may be optimistic because the same folds selected K.

Charts: roc_curves.png, roc_auc_by_fold.png, roc_auc_by_k.png. Numeric outputs: roc_auc_by_class.csv, roc_curves.csv, fold_metrics.csv, tuning.csv, and metrics.json. predictions.csv includes the selected K's out-of-fold class probabilities.

See training_balance.csv for exact class counts and knn_tuning.png for the tuning curve. KNN and neural network scores use different targets and split recipes and should not be compared directly.
