# ads2026

Wine quality exploration and **Low / Medium / High classification** with KNN
and a neural network. The active workflow is `preprocessing_restratified.py`.

```bash
python3 -m pip install -r requirements-neural-network.txt
python3 preprocessing_restratified.py
```

Run just the neural network with either command:

```bash
python3 preprocessing_restratified.py --model neural-network
python3 train_wine_neural_network.py
```

Run just KNN with `python3 preprocessing_restratified.py --model knn`
or `python3 model_training.py`. Both use K=3.
Outputs are in [wine_restratified_visuals/](wine_restratified_visuals/README.md).

Both models predict the existing `quality_class` labels: **Low, Medium, High**.
They use 11 chemical measurements and red/white wine type as inputs. Numerical
`quality` is unused by training and is not required in a training CSV.
Neither `quality` nor `quality_class` is an input feature. Source data and
historical exploratory charts retain the original numerical score.

The neural network has three output logits and trains with cross-entropy.
Softmax produces probabilities in Low/Medium/High order, and the class with
the largest probability is the prediction. Both candidate architectures
(32–16 and 64–32 hidden units) are selected by mean validation macro F1.
There is no numerical score prediction or score rounding.

Training partitions preserve observed class proportions. Each source training
row is used once per epoch or fit, without resampling or class weights.
StandardScaler is fitted only on the corresponding training rows.

The neural network reserves a stratified 10% holdout and uses five outer
validation folds by default. An inner approximately 80/20 stratified split
selects an epoch count by maximum macro F1, then a fresh network refits the
outer training rows. Identical inputs stay together; duplicate feature/label
rows are removed. The final development fit uses the rounded median of the
selected architecture's fold epoch counts.

KNN keeps a stratified 80/20 holdout, ten validation folds, Manhattan distance,
distance weighting, and **K fixed at 3** as specified. There is no search over K.
Training entry points leave reserved test rows unscored. Neural network validation
selects its architecture; KNN uses the fixed K=3. These are validation results. The two
models use different partitions and do not provide a paired comparison.

The [neural network report](wine_restratified_visuals/wine_neural_network/report.md)
contains accuracy, macro F1, ROC AUC, and baseline comparisons. Its output folder
includes:

- `predictions.csv`: out-of-fold actual/predicted classes and three probabilities.
- `cv_predictions.csv`: predictions for both architectures and classifier baselines.
- `validation_metrics_by_class.csv` and `confusion_matrix.csv`: per-class results.
- `roc_curves.png/.csv`, `roc_auc_by_class.csv`, and `roc_auc_by_fold.png`: ROC results.
- Learning curves, validation plots, and training/split class distributions.
- `model.pt` and `scaler.joblib`: the final classifier, class order, and scaler.
- Fold assignments, histories, partition memberships, and training-row counts.

Evaluate the saved neural network on its remaining reserved 10% with:

```bash
python3 testing.py
```

This loads `model.pt`, `scaler.joblib`, and the original test membership from
`partition_assignments.csv`. It checks the source CSV checksum, keeps training
and test rows separate, and performs inference without refitting. Numerical
quality is unused. Results are saved in
[wine_neural_network/testing/](wine_restratified_visuals/wine_neural_network/testing/report.md):
`test_accuracy.png` (accuracy and confusion matrix), `test_roc_curves.png`,
`test_metrics.json`, per-class metrics, and test predictions. Options are
`--csv`, `--model-dir`, and `--output`.

Optional arguments: `--model`, `--folds`, `--epochs` (maximum inner-fit epochs),
`--patience`, `--csv`, and `--output`. With `--model all`, `--output` names the
parent of the two model folders. Each report records the source CSV checksum.

`preprocessing.py` and the original top-level output
folders are historical references to the earlier balanced experiment. They
are not the active workflow. `train_wine_neural_network.py` now runs the active
classifier.

Run classifier checks with `python3 -m unittest discover -s tests`.
