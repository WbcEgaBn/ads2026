# ads2026

Wine quality exploration and **Low / Medium / High classification** with KNN,
a neural network, and a random forest. `preprocessing_restratified.py` trains
KNN and the neural network; the forest reuses the saved neural network splits.

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

All three models predict the existing `quality_class` labels: **Low, Medium, High**.
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

Train and validate the random forest after the neural network artifacts exist:

```bash
python3 train_wine_random_forest.py
```

This does **not** predict or score test rows. Evaluate the saved forest only through:

```bash
python3 testing.py --model random-forest
```

It reads the neural checkpoint's metadata and reuses the exact development/test
row IDs, validation folds, feature order, class encoding, seed, and final scaler.
The forest learns its own trees; neural network weights and settings such as
hidden layers, learning rate, and epochs do not apply. It uses 300 trees with
`max_features="sqrt"`, no class weights, and `bootstrap=False` to retain every
training row in each tree. Parameters are fixed before validation and testing.

The [random forest report](wine_restratified_visuals/wine_random_forest/report.md)
links matching validation ROC curves, confusion matrices, training/validation
performance, model comparisons, fold metrics, class distributions, and learning
curves by number of trees. It also includes feature importance and separate
test accuracy/ROC plots under `wine_random_forest/testing/` when `testing.py` is run. Validation comparisons
use the selected neural network and its baselines on the same saved folds.
`model.joblib` stores the fitted scaler/forest pipeline; output classes use
Low=0, Medium=1, High=2. `aligned_probabilities(model, X)` from the forest script
returns probabilities in that order. Neural artifacts are preserved.

Forest options: `--csv`, `--neural-dir`, `--output`, `--trees`, and `--jobs`.
`testing.py` defaults to the neural network; `--model random-forest` selects the
forest and its output directory. No training entry point invokes test evaluation.

Optional arguments: `--model`, `--folds`, `--epochs` (maximum inner-fit epochs),
`--patience`, `--csv`, and `--output`. With `--model all`, `--output` names the
parent of the two model folders. Each report records the source CSV checksum.

`preprocessing.py` and the original top-level output
folders are historical references to the earlier balanced experiment. They
are not the active workflow. `train_wine_neural_network.py` now runs the active
classifier.

Run classifier checks with `python3 -m unittest discover -s tests`.
