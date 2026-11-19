# Random forest Low / Medium / High classifier

The neural network's model.pt supplies source metadata and class encoding, not learned weights.
Inputs are the same 11 chemical measurements and red/white indicator. Numerical quality is unused.
Reuses the exact 4,788 development rows, 5 validation folds, and 532 reserved test rows.
Fold scalers use only their training rows; the final fit reuses the neural network's saved development scaler.
Fixed forest parameters: {'n_estimators': 300, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': False, 'class_weight': None, 'random_state': 2026, 'n_jobs': -1}. There is no search or test-based model selection.
No class weighting, class resampling, or per-tree bootstrapping is used. Neural hidden layers, epochs, and learning rates do not apply.

| Split | Accuracy | Macro F1 | Macro ROC AUC |
|---|---:|---:|---:|
| Out-of-fold validation | 79.6% | 0.5005 | 0.8249 |

The reserved test rows are not predicted or scored by this training script.
Evaluate them separately with `python3 testing.py --model random-forest`.

| Model, identical validation folds | Mean accuracy | Mean macro F1 | Mean macro ROC AUC |
|---|---:|---:|---:|
| Random forest | 79.6% | 0.5003 | 0.8254 |
| Neural network 64-32 | 79.4% | 0.4958 | 0.8094 |
| Majority baseline | 76.6% | 0.2891 | 0.5000 |
| Logistic regression | 78.2% | 0.4430 | 0.7913 |

Learning curves add trees on each outer training fold. The final tree count is fixed in advance.
ROC curves use aligned Low/Medium/High probabilities, one class versus the rest.
Final-model training scores describe a different fit from the out-of-fold validation models.
These source rows and test results have been used in earlier experiments; this is not a new unseen dataset.

Saved model: model.joblib is a fitted scaler/forest pipeline; predict returns Low=0, Medium=1, High=2.
aligned_probabilities(model, X) returns probabilities in Low/Medium/High order. features and classes are in metrics.json.
The neural checkpoint and scaler are preserved; the forest is stored separately.

- [Validation ROC](roc_curves.png), [confusion matrices](validation_predictions.png), [training/validation comparison](split_performance.png)
- [Learning curves](learning_curves.png), [model comparison](baseline_comparison.png), [fold performance](cross_validation.png)
- [ROC by fold](roc_auc_by_fold.png), [training classes](training_class_distribution.png), [partition classes](split_class_distribution.png)
- [Feature importance](feature_importance.png)

Reproduce: `python3 train_wine_random_forest.py`.
