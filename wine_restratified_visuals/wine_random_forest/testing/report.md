# Random forest reserved-test evaluation

Evaluated **532 reserved rows** with the saved 300-tree random forest and scaler.
The exact test membership comes from partition_assignments.csv. No model or scaler was refitted.
Predictions are Low, Medium, or High. Numerical quality is unused.

Accuracy: **82.5%**; macro F1: **0.5430**; macro ROC AUC: **0.8248**.
The training-majority baseline always predicts Medium: test accuracy 76.7%.

| Class | Precision | Recall | F1 | Test rows |
|---|---:|---:|---:|---:|
| Low | 50.0% | 8.7% | 0.1481 | 23 |
| Medium | 84.3% | 94.9% | 0.8927 | 408 |
| High | 72.5% | 49.5% | 0.5882 | 101 |

ROC curves use class probabilities with each class versus the rest.
This evaluates the reserved portion of this training run. These source rows overlap earlier experiments.

- [Test accuracy and confusion matrix](test_accuracy.png)
- [Test ROC curves](test_roc_curves.png)
- [Metrics](test_metrics.json) and [per-row predictions](test_predictions.csv)

Reproduce: `python3 testing.py --model random-forest`.
