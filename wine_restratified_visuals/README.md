# Restratified wine analysis

Run the complete workflow from the repository root:

```bash
python3 preprocessing_restratified.py
```

This is the active entry point for this experiment. The legacy training code in preprocessing.py
and its compatibility commands are retained as inactive reference code for this experiment.
Existing wine_visuals/, wine_knn/, and wine_neural_network/ outputs are preserved.

Restratification preserves the observed Low/Medium/High proportions in each split.
It does not force equal class totals. No training row is duplicated or discarded for class balancing,
and neither model uses class weights. Each neural network epoch traverses its original training rows once.
Scaling and target centering use training rows only. quality_class is excluded from model features.

KNN retains its stratified 80/20 holdout, 10 validation folds, Manhattan distance, distance weighting,
and odd candidate K values from 1 through 21. It selects K by mean validation macro F1.
The neural network uses a newly stratified 10% holdout, 5 outer validation folds, and an inner
approximately 80/20 stratified split for early stopping. Identical inputs stay in the same partition.
Exact stratification uses StratifiedKFold when every input group is a singleton; otherwise
StratifiedGroupKFold approximates the class proportions while preserving groups.
Exact duplicate removal in the neural network retains its legacy behavior; none were found in this dataset.

Reserved tests are not predicted or evaluated. These source rows overlap earlier experiments.
Validation also selects the models, so these scores are not independent test estimates.
KNN classification and neural network regression use different targets and split recipes.

Source: `wine2.csv`; SHA-256: `ff8898e24f19f48a77ffe8f5a7a0d2784388f184f4e4a4bf3276841eeebaacff`.

Generated 34 charts in this separate output tree:

- [exploratory/](exploratory/): regenerated source-data charts.
- [wine_knn/report.md](wine_knn/report.md): KNN tuning, validation, ROC AUC, and class distributions.
- [wine_neural_network/report.md](wine_neural_network/report.md): network, baselines, learning curves, and validation.

KNN: K=1; mean validation macro F1 0.5281; pooled validation macro ROC AUC 0.6462.
Neural network: [32, 16]; mean validation MAE 0.5418; exact rounded accuracy 56.0%; final fit 58 epochs.

Each model directory includes training_rows.csv (unique training counts), partition_assignments.csv
(source-row membership), and split_class_distribution.csv/.png (class counts and proportions).
Reserved row IDs appear only as membership records, without test predictions or scores.
