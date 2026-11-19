# Neural network reserved-test evaluation

Evaluated **532 reserved rows** with the saved [64, 32] classifier and scaler.
The exact test membership comes from partition_assignments.csv. No model or scaler was refitted.
Predictions are Low, Medium, or High. Numerical quality is unused.

Accuracy: **80.5%**; macro F1: **0.4487**; macro ROC AUC: **0.8251**.
The training-majority baseline always predicts Medium: test accuracy 76.7%.

| Class | Precision | Recall | F1 | Test rows |
|---|---:|---:|---:|---:|
| Low | 0.0% | 0.0% | 0.0000 | 23 |
| Medium | 81.4% | 96.6% | 0.8834 | 408 |
| High | 73.9% | 33.7% | 0.4626 | 101 |

ROC curves use softmax probabilities with each class versus the rest.
This evaluates the reserved portion of this training run. These source rows overlap earlier experiments.

- [Test accuracy and confusion matrix](test_accuracy.png)
- [Test ROC curves](test_roc_curves.png)
- [Metrics](test_metrics.json) and [per-row predictions](test_predictions.csv)

Reproduce: `python3 testing.py` (defaults to the active saved neural network).
