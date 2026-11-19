# Restratified wine analysis

Run the complete workflow from the repository root:

```bash
python3 preprocessing_restratified.py
```

This is the active entry point. model_training.py runs KNN with K=3; train_wine_neural_network.py runs the neural classifier.
The legacy preprocessing.py is an inactive reference to the balanced experiment.
Existing wine_visuals/, wine_knn/, and wine_neural_network/ outputs are preserved.

Restratification preserves the observed Low/Medium/High proportions in each split.
It does not force equal class totals. No training row is duplicated or discarded for class balancing,
and neither model uses class weights. Each neural network epoch traverses its original training rows once.
Scaling uses training rows only. Both quality and quality_class are excluded from model features.
Both models predict Low, Medium, or High. Numerical quality is unused by training and is not required.

KNN retains its stratified 80/20 holdout, 10 validation folds, Manhattan distance, distance weighting,
and K fixed at 3 as specified. No search over K is performed.
The neural network uses a newly stratified 10% holdout, 5 outer validation folds, and an inner
approximately 80/20 stratified split for early stopping. Identical inputs stay in the same partition.
Exact stratification uses StratifiedKFold when every input group is a singleton; otherwise
StratifiedGroupKFold approximates the class proportions while preserving groups.
Duplicate feature/label rows are removed before neural network splitting; numerical quality is ignored.

Training entry points do not evaluate reserved tests. Run testing.py to evaluate the saved neural network.
Neural network validation selects its architecture; KNN uses the specified K=3. Source rows overlap earlier experiments.
Both models are classifiers with different split recipes; validation scores are not a paired comparison.

Source: `wine2.csv`; SHA-256: `ff8898e24f19f48a77ffe8f5a7a0d2784388f184f4e4a4bf3276841eeebaacff`.

Generated 49 charts in this separate output tree:

- [exploratory/](exploratory/): regenerated source-data charts.
- [wine_knn/report.md](wine_knn/report.md): KNN K=3 validation, ROC AUC, and class distributions.
- [wine_neural_network/report.md](wine_neural_network/report.md): network, baselines, learning curves, and validation.

KNN: K=3; mean validation macro F1 0.5139; pooled validation macro ROC AUC 0.7236.
Neural network: [64, 32]; mean validation macro F1 0.4958; accuracy 79.4%; final fit 73 epochs.

Each model directory includes training_rows.csv (unique training counts), partition_assignments.csv
(source-row membership), and split_class_distribution.csv/.png (class counts and proportions).
Training outputs contain reserved row IDs as membership records. Neural network test results are saved separately in testing/.

[Neural network reserved-test results](wine_neural_network/testing/report.md): 532 rows; accuracy 80.5%; macro F1 0.4487; macro ROC AUC 0.8251.

[Random forest on the same neural network splits](wine_random_forest/report.md). Run `python3 train_wine_random_forest.py` after neural network training.
Validation accuracy 79.6%; macro F1 0.5005; macro ROC AUC 0.8249.
Training leaves test rows unscored. Evaluate separately with `python3 testing.py --model random-forest`.
[Random forest testing.py results](wine_random_forest/testing/report.md): accuracy 82.5%; macro F1 0.5430; macro ROC AUC 0.8248.
