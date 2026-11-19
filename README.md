# ads2026

Wine quality exploration, KNN classification, and neural network regression
using `Combined_Wine.csv`. Both training implementations and their shared
preprocessing live in `preprocessing.py`.

Install dependencies and regenerate the outputs:

```bash
python3 -m pip install -r requirements-neural-network.txt
python3 wine_visuals.py
python3 preprocessing.py
```

Run one model with `python3 preprocessing.py --model knn` or
`python3 preprocessing.py --model neural-network --folds 5`.
The original `model_training.py` and `train_wine_neural_network.py` commands
remain available as thin entry points.

Both models fit `StandardScaler` on the original training rows of each fold,
then use the same class balancing function: retain all Medium training rows
and resample Low and High with replacement to match the Medium count.
Every neural network epoch sees those exact equal class totals. Inner training,
outer fold refits, and the final neural network fit all follow this rule.
Validation rows retain their original class proportions. Reserved test rows
are neither transformed nor evaluated.

The neural network workflow trains and validates only. It reserves the original
10% test partition without predicting or scoring it. Grouped k-fold validation
runs on the remaining 90%; identical input measurements stay in one fold.
Each fold uses a separate inner validation split to select its training epoch
count, then refits on the outer training fold. Scaling uses training rows only.
The architecture with the lowest mean fold MAE is trained on all development
rows for the median selected epoch count, rounded to the nearest integer.

Model outputs and validation charts are in `wine_neural_network/`; see
[`report.md`](wine_neural_network/report.md) for results and limitations.
`predictions.csv` contains selected-model out-of-fold validation predictions;
`cv_predictions.csv` includes both architectures and the baselines.
`fold_metrics.csv` and `fold_assignments.csv` record the validation folds.
`model.pt` and `scaler.joblib` contain the final development-data fit.
`training_balance.csv` records class counts before and after balancing for
every training fit; `training_class_balance.png` shows the final fit's counts.

KNN keeps its existing stratified 80/20 split, all source rows, 10-fold default,
distance weighting, Manhattan distance, and macro F1 tuning over odd K values
from 1 through 21. Its scaler is now fitted separately within each fold.
Outputs are saved in `wine_knn/`. The neural network retains numerical quality
as its target and balances its training samples using `quality_class` labels;
the class label is excluded from the input features. The models' different
targets and data splits mean their validation scores cannot be compared directly.

Exploratory charts are in `wine_visuals/`. They describe all source rows,
including duplicates; model fitting removes exact duplicate rows.

Optional training arguments: `--model`, `--folds`, `--epochs` (maximum per inner
neural network fit), `--patience`, `--csv`, and `--output`.
With `--model all` (the default), `--output` is the parent of the two model
directories and `--folds` overrides both models' defaults.
