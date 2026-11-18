# Wine quality neural network

Target: numerical quality score. Inputs: 11 chemical measurements plus red/white wine type.
Removed 1,177 identical rows from 6,497 source rows.
70/20/10 train/validation/test split sizes: {'train': 3724, 'validation': 1064, 'test': 532}. Identical inputs cannot cross splits.
The scaler and target mean are fitted on training data only. quality_class is excluded to avoid target leakage.
Selected hidden layers: (64, 32); selected epoch: 81. Selection uses validation MAE only.
The test data is evaluated after architecture and checkpoint selection. Models are not refitted on validation data.

| Model | Test MAE | Test RMSE | Test R² | Exact rounded score | Within one rounded point |
|---|---:|---:|---:|---:|---:|
| Neural network | 0.516 | 0.661 | 0.392 | 58.5% | 96.8% |
| Mean baseline | 0.681 | 0.847 | -0.001 | 43.4% | 94.4% |
| Ridge regression | 0.531 | 0.683 | 0.350 | 56.0% | 96.1% |

MAE and RMSE use continuous predictions. Accuracy rounds to the nearest integer (half-up), clipped to the 0–10 quality scale.
Within-one accuracy uses the rounded prediction, not a continuous-error threshold.
Scores near 5 and 6 dominate; aggregate accuracy can hide weak performance on rare scores. See test_metrics_by_quality.csv.
This is a single fixed random split; results may vary with other seeds and do not establish performance on a different wine population.

Reproduce: `python3 train_wine_neural_network.py`

Plots: learning_curves.png, split_performance.png, test_predictions.png, baseline_comparison.png.
Saved model: model.pt; preprocessing: scaler.joblib. Predictions and split assignments: predictions.csv.
