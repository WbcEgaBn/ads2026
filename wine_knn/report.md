# KNN training and validation

Uses the original raw rows and stratified 80/20 development/test split. No test evaluation is performed.

StandardScaler is fitted separately on each training fold. Medium training rows are retained; Low and High are resampled with replacement to match Medium. The neural network uses the same balancing helper in preprocessing.py. Validation rows retain their original distribution.

Selected K: 1; mean 10-fold macro F1: 0.6003 ± 0.0379. This score also selects K and is not an independent test estimate.

See training_balance.csv for exact class counts and knn_tuning.png for the tuning curve. KNN and neural network scores use different targets and split recipes and should not be compared directly.
