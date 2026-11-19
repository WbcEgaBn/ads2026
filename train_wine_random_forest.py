"""Train a Low/Medium/High random forest on the neural network's saved splits.

Run: python3 train_wine_random_forest.py
The neural checkpoint supplies metadata, not weights. Forest parameters are
fixed before validation. This script trains and validates only. Run
testing.py --model random-forest for reserved-test evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

import preprocessing_restratified as shared


NAME = "Random forest"
DEFAULT_TREES = 300


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_probabilities(model, X):
    """Targets use the neural network's Low=0, Medium=1, High=2 encoding."""
    order = [list(model.classes_).index(i) for i in range(len(shared.CLASSES))]
    return model.predict_proba(X)[:, order]


def prediction_frame(raw, indices, probabilities, split, fold=None):
    frame = raw.iloc[indices][["wine_type", "quality_class"]].rename(
        columns={"quality_class": "actual_class"}).reset_index(drop=True)
    frame.insert(0, "source_row", indices)
    frame["predicted_class"] = np.asarray(shared.CLASSES)[probabilities.argmax(axis=1)]
    frame["model"], frame["split"] = NAME, split
    if fold is not None:
        frame["fold"] = fold
    for i, column in enumerate(shared.PROBABILITY_COLUMNS):
        frame[column] = probabilities[:, i]
    return frame


def load_reference(csv, neural_dir):
    """Use the saved source IDs, including the exact five outer-fold assignments."""
    checkpoint = torch.load(neural_dir / "model.pt", map_location="cpu", weights_only=True)
    metadata = json.loads((neural_dir / "metrics.json").read_text())
    if (checkpoint.get("task") != "multiclass_classification"
            or checkpoint.get("classes") != shared.CLASSES or checkpoint.get("features") != shared.FEATURES
            or checkpoint["data_source"]["sha256"] != shared.data_source(csv)["sha256"]):
        raise ValueError("The neural checkpoint must match this source and its Low/Medium/High features.")
    raw, X, development, reserved = shared.load_saved_holdout(csv, neural_dir, metadata)
    scaler = joblib.load(neural_dir / "scaler.joblib")
    if (checkpoint["training_rows"] != len(development)
            or list(scaler.feature_names_in_) != shared.FEATURES
            or np.any(scaler.n_samples_seen_ != len(development))):
        raise ValueError("The neural scaler and checkpoint must match the saved development rows.")
    partitions = pd.read_csv(neural_dir / "partition_assignments.csv")
    assignments = pd.read_csv(neural_dir / "fold_assignments.csv")
    if (assignments.source_row.duplicated().any() or set(assignments.source_row) != set(development)
            or set(assignments.fold) != set(range(1, metadata["folds"] + 1))):
        raise ValueError("Neural network validation folds must cover each development row exactly once.")
    y = pd.Categorical(raw.quality_class, categories=shared.CLASSES).codes.astype(np.int64)
    if (y < 0).any():
        raise ValueError("Targets must be Low, Medium, or High.")
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    for fold, part in assignments.groupby("fold"):
        validation = part.source_row.to_numpy()
        training = development[~np.isin(development, validation)]
        recorded = partitions[partitions.stage == f"outer_fold_{fold}"]
        if (set(recorded.loc[recorded.split == "train", "source_row"]) != set(training)
                or set(recorded.loc[recorded.split == "validation", "source_row"]) != set(validation)):
            raise ValueError("Fold assignments do not match the saved neural network partitions.")
        if set(groups[training]) & set(groups[validation]):
            raise ValueError("Identical inputs cross a training/validation boundary.")
    return raw, X, y, development, reserved, assignments, partitions, scaler, checkpoint, metadata


def class_metrics(predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions.actual_class, predictions.predicted_class, labels=shared.CLASSES, zero_division=0)
    return pd.DataFrame({"class": shared.CLASSES, "precision": precision,
                         "recall": recall, "f1": f1, "support": support})


def score_predictions(predictions):
    codes = pd.Categorical(predictions.actual_class, categories=shared.CLASSES).codes
    return shared.metrics(codes, predictions[shared.PROBABILITY_COLUMNS].to_numpy())


def reference_comparison(neural_dir, metadata, assignments, raw):
    """Reuse existing comparator predictions from these exact folds; no refits."""
    names = [metadata["selected_model"], "Majority baseline", "Logistic regression"]
    predictions = pd.read_csv(neural_dir / "cv_predictions.csv")
    predictions = predictions[predictions.model.isin(names)].copy()
    if set(predictions.model) != set(names):
        raise ValueError("Saved neural network and baseline predictions are missing.")
    rows = []
    expected = assignments.set_index("source_row").fold.sort_index()
    for name, part in predictions.groupby("model", sort=False):
        if part.source_row.duplicated().any() or set(part.source_row) != set(expected.index):
            raise ValueError("Comparator predictions must cover the same development rows exactly once.")
        actual = part.set_index("source_row").fold.sort_index()
        if not actual.equals(expected):
            raise ValueError("Comparator predictions use different validation folds.")
        if not np.array_equal(raw.iloc[part.source_row].quality_class.to_numpy(), part.actual_class):
            raise ValueError("Comparator labels do not match the source rows.")
        for fold, fold_part in part.groupby("fold"):
            rows.append({"model": name, "fold": fold, **score_predictions(fold_part)})
    return predictions, pd.DataFrame(rows)


def make_plots(history, fold_metrics, comparison, training_scores, validation_scores, predictions, output):
    plt = shared.plt
    plt.style.use("seaborn-v0_8-whitegrid")
    folds = sorted(fold_metrics.fold.unique())
    fig, axes = plt.subplots(len(folds), 2, figsize=(12, 2.7 * len(folds)), squeeze=False)
    for fold, pair in zip(folds, axes):
        part = history[history.fold == fold]
        for split, color in [("train", "#2166ac"), ("validation", "#b35806")]:
            for ax, key in zip(pair, ["log_loss", "macro_f1"]):
                ax.plot(part.trees, part[f"{split}_{key}"], marker="o", color=color, label=split.capitalize())
                ax.set_xlabel("Number of trees")
                ax.legend(fontsize=8)
        pair[0].set_ylabel(f"Fold {fold}: log loss")
        pair[1].set(ylabel="Macro F1", ylim=(0, 1))
    fig.suptitle("Random forest learning curves\nTree count is fixed in advance; curves do not select a model", y=1.002)
    shared.save_plot(fig, output, "learning_curves.png")

    keys = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("roc_auc_macro", "Macro ROC AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (key, title) in zip(axes, keys):
        values = [training_scores[key], validation_scores[key]]
        bars = ax.bar(["Final model\ntraining", "Out-of-fold\nvalidation"], values, color=["#2166ac", "#d98f39"])
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4)
        ax.set(title=title, ylim=(0, 1.12))
    fig.suptitle("Random forest training and validation (different fitted models)")
    shared.save_plot(fig, output, "split_performance.png")

    matrix = confusion_matrix(predictions.actual_class, predictions.predicted_class, labels=shared.CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, normalized in zip(axes, [False, True]):
        values = matrix / matrix.sum(axis=1, keepdims=True) if normalized else matrix
        img = ax.imshow(values, cmap="Blues", vmin=0)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{values[i, j]:.1%}" if normalized else str(matrix[i, j]),
                        ha="center", va="center", color="white" if values[i, j] > values.max() / 2 else "black")
        ax.set(xticks=range(3), xticklabels=shared.CLASSES, yticks=range(3), yticklabels=shared.CLASSES,
               xlabel="Predicted class", ylabel="Actual class", title="Recall by actual class" if normalized else "Validation counts")
        ax.grid(False)
        fig.colorbar(img, ax=ax, shrink=0.7)
    fig.suptitle("Random forest: out-of-fold validation predictions")
    shared.save_plot(fig, output, "validation_predictions.png")
    pd.DataFrame(matrix, index=pd.Index(shared.CLASSES, name="actual_class"), columns=shared.CLASSES).to_csv(output / "confusion_matrix.csv")

    names = list(comparison.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axes, keys):
        means = [comparison.loc[comparison.model == name, key].mean() for name in names]
        std = [comparison.loc[comparison.model == name, key].std(ddof=1) for name in names]
        bars = ax.bar(names, means, yerr=std, capsize=4, color=["#37966f", "#2166ac", "#999999", "#648db1"])
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=8)
        ax.set(title=title, ylim=(0, 1.15))
        ax.tick_params(axis="x", labelsize=8, labelrotation=15)
    fig.suptitle("Models evaluated on identical validation folds (error bars: ±1 fold SD)")
    shared.save_plot(fig, output, "baseline_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in names:
        part = comparison[comparison.model == name].sort_values("fold")
        for ax, key in zip(axes, ["macro_f1", "accuracy"]):
            ax.plot(part.fold, part[key], marker="o", label=name)
    for ax, label in zip(axes, ["Macro F1", "Accuracy"]):
        ax.set(xlabel="Validation fold", ylabel=label, xticks=folds, ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle("Classification performance on matching validation folds")
    shared.save_plot(fig, output, "cross_validation.png")


def run(args):
    neural_dir, output = Path(args.neural_dir), Path(args.output)
    if args.trees < 1:
        raise ValueError("trees must be positive")
    if output.resolve() == neural_dir.resolve():
        raise ValueError("Choose a separate random forest output directory.")
    (raw, X, y, development, reserved, assignments, partitions, scaler,
     checkpoint, neural_metrics) = load_reference(args.csv, neural_dir)
    reference_predictions, reference_scores = reference_comparison(neural_dir, neural_metrics, assignments, raw)
    reference_hashes = {name: file_hash(neural_dir / name)
                        for name in ["model.pt", "scaler.joblib", "partition_assignments.csv", "fold_assignments.csv"]}
    params = {"n_estimators": args.trees, "max_features": "sqrt", "max_depth": None,
              "min_samples_split": 2, "min_samples_leaf": 1, "bootstrap": False,
              "class_weight": None, "random_state": neural_metrics["seed"], "n_jobs": args.jobs}
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(set(min(trees, args.trees) for trees in [25, 50, 100, 200, args.trees]))
    histories, scores, parts, audit = [], [], [], []
    for fold, part in assignments.groupby("fold", sort=True):
        validation = part.source_row.to_numpy()
        train = development[~np.isin(development, validation)]
        fold_scaler = shared.StandardScaler().fit(X.iloc[train])
        train_X, val_X = fold_scaler.transform(X.iloc[train]), fold_scaler.transform(X.iloc[validation])
        forest = RandomForestClassifier(**{**params, "n_estimators": checkpoints[0]}, warm_start=True)
        shared.record_training(audit, f"outer_fold_{fold}", NAME, train, raw.quality_class.to_numpy())
        for trees in checkpoints:
            forest.set_params(n_estimators=trees)
            forest.fit(train_X, y[train])
            train_scores = shared.metrics(y[train], aligned_probabilities(forest, train_X))
            probabilities = aligned_probabilities(forest, val_X)
            val_scores = shared.metrics(y[validation], probabilities)
            histories.append({"fold": fold, "trees": trees,
                              **{f"train_{key}": train_scores[key] for key in shared.METRIC_KEYS},
                              **{f"validation_{key}": val_scores[key] for key in shared.METRIC_KEYS}})
        scores.append({"model": NAME, "fold": fold, "train_n": len(train), **val_scores})
        parts.append(prediction_frame(raw, validation, probabilities, "validation", fold))
        print(f"Random forest fold {fold}/{neural_metrics['folds']}: macro F1 {val_scores['macro_f1']:.4f}", flush=True)
    predictions = pd.concat(parts, ignore_index=True).sort_values("source_row")
    if set(predictions.source_row) != set(development) or predictions.source_row.duplicated().any():
        raise ValueError("Out-of-fold predictions must cover exactly the development rows.")
    fold_metrics, history = pd.DataFrame(scores), pd.DataFrame(histories)
    comparison = pd.concat([fold_metrics, reference_scores], ignore_index=True)
    forest = RandomForestClassifier(**params)
    forest.fit(scaler.transform(X.iloc[development]), y[development])
    pipeline = Pipeline([("scaler", scaler), ("classifier", forest)])
    shared.record_training(audit, "final", NAME, development, raw.quality_class.to_numpy())
    train_scores = shared.metrics(y[development], aligned_probabilities(pipeline, X.iloc[development]))
    pooled = score_predictions(predictions)
    summary = {"model": NAME, "task": "multiclass_classification", "target": "quality_class",
               "features": shared.FEATURES, "classes": shared.CLASSES, "class_encoding": dict(enumerate(shared.CLASSES)),
               "data_source": shared.data_source(args.csv), "seed": neural_metrics["seed"], "parameters": params,
               "parameter_selection": "Fixed before validation/test evaluation; no hyperparameter search",
               "reference_neural_artifact_hashes": reference_hashes,
               "split_sizes": {"development": len(development), "test_reserved": len(reserved)},
               "training_rows": len(development), "test_evaluated": False,
               "folds": neural_metrics["folds"], "same_neural_network_folds": True,
               "training_policy": "No class weighting or resampling; bootstrap=False uses all training rows per tree",
               "scaling": "Training-only scaler per fold; saved neural development scaler reused for final fit",
               "validation_metrics": pooled, "final_training_metrics": train_scores,
               "mean_fold_metrics": {key: float(fold_metrics[key].mean()) for key in shared.METRIC_KEYS},
               "std_fold_metrics": {key: float(fold_metrics[key].std(ddof=1)) for key in shared.METRIC_KEYS},
               "final_training_class_counts": {label: audit[-1][label] for label in shared.CLASSES}}
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.concat([predictions, reference_predictions], ignore_index=True).to_csv(output / "cv_predictions.csv", index=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    comparison.to_csv(output / "comparison_fold_metrics.csv", index=False)
    assignments.to_csv(output / "fold_assignments.csv", index=False)
    history.to_csv(output / "learning_history.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "training_rows.csv", index=False)
    class_metrics(predictions).to_csv(output / "validation_metrics_by_class.csv", index=False)
    pd.DataFrame([{"wine_type": key, **score_predictions(part)} for key, part in predictions.groupby("wine_type")]).to_csv(
        output / "validation_metrics_by_wine_type.csv", index=False)
    joblib.dump(pipeline, output / "model.joblib", compress=3)
    joblib.dump(scaler, output / "scaler.joblib")
    summary["model_sha256"] = file_hash(output / "model.joblib")
    summary["roc_auc"] = shared.save_neural_roc_outputs(predictions, fold_metrics, output, model_name=NAME)
    make_plots(history, fold_metrics, comparison, train_scores, pooled, predictions, output)
    plot_partitions = []
    for (stage, split), part in partitions.groupby(["stage", "split"], sort=False):
        if stage == "holdout" or stage.startswith("outer_fold_"):
            plot_partitions.append((stage, split, part.source_row.to_numpy()))
    shared.save_partition_audit(plot_partitions, raw.quality_class.to_numpy(), np.arange(len(raw)), output)
    shared.plot_training_distribution(audit, output)
    importance = pd.DataFrame({"feature": shared.FEATURES, "importance": forest.feature_importances_}).sort_values("importance")
    importance.to_csv(output / "feature_importance.csv", index=False)
    fig, ax = shared.plt.subplots(figsize=(9, 6))
    ax.barh(importance.feature, importance.importance, color="#347d58")
    ax.set(xlabel="Mean decrease in impurity (training)", title="Random forest feature importance")
    shared.save_plot(fig, output, "feature_importance.png")

    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_report(summary, comparison, output)
    for name, digest in reference_hashes.items():
        if file_hash(neural_dir / name) != digest:
            raise RuntimeError(f"The reference neural artifact changed during this run: {name}")
    print(f"Random forest validation: accuracy {pooled['accuracy']:.1%}; macro F1 {pooled['macro_f1']:.4f}; "
          f"reserved test rows {len(reserved)} not evaluated.", flush=True)
    print(f"Saved model, reports, and visuals to {output.resolve()}")
    return summary


def write_report(summary, comparison, output):
    validation = summary["validation_metrics"]
    lines = ["# Random forest Low / Medium / High classifier", "",
             "The neural network's model.pt supplies source metadata and class encoding, not learned weights.",
             "Inputs are the same 11 chemical measurements and red/white indicator. Numerical quality is unused.",
             f"Reuses the exact {summary['split_sizes']['development']:,} development rows, {summary['folds']} validation folds, and {summary['split_sizes']['test_reserved']} reserved test rows.",
             "Fold scalers use only their training rows; the final fit reuses the neural network's saved development scaler.",
             f"Fixed forest parameters: {summary['parameters']}. There is no search or test-based model selection.",
             "No class weighting, class resampling, or per-tree bootstrapping is used. Neural hidden layers, epochs, and learning rates do not apply.", "",
             "| Split | Accuracy | Macro F1 | Macro ROC AUC |", "|---|---:|---:|---:|",
             f"| Out-of-fold validation | {validation['accuracy']:.1%} | {validation['macro_f1']:.4f} | {validation['roc_auc_macro']:.4f} |",
             "", "The reserved test rows are not predicted or scored by this training script.",
             "Evaluate them separately with `python3 testing.py --model random-forest`.", "",
             "| Model, identical validation folds | Mean accuracy | Mean macro F1 | Mean macro ROC AUC |",
             "|---|---:|---:|---:|"]
    for name, part in comparison.groupby("model", sort=False):
        lines.append(f"| {name} | {part.accuracy.mean():.1%} | {part.macro_f1.mean():.4f} | {part.roc_auc_macro.mean():.4f} |")
    lines.extend(["", "Learning curves add trees on each outer training fold. The final tree count is fixed in advance.",
                  "ROC curves use aligned Low/Medium/High probabilities, one class versus the rest.",
                  "Final-model training scores describe a different fit from the out-of-fold validation models.",
                  "These source rows and test results have been used in earlier experiments; this is not a new unseen dataset.", "",
                  "Saved model: model.joblib is a fitted scaler/forest pipeline; predict returns Low=0, Medium=1, High=2.",
                  "aligned_probabilities(model, X) returns probabilities in Low/Medium/High order. features and classes are in metrics.json.",
                  "The neural checkpoint and scaler are preserved; the forest is stored separately.", "",
                  "- [Validation ROC](roc_curves.png), [confusion matrices](validation_predictions.png), [training/validation comparison](split_performance.png)",
                  "- [Learning curves](learning_curves.png), [model comparison](baseline_comparison.png), [fold performance](cross_validation.png)",
                  "- [ROC by fold](roc_auc_by_fold.png), [training classes](training_class_distribution.png), [partition classes](split_class_distribution.png)",
                  "- [Feature importance](feature_importance.png)", "",
                  "Reproduce: `python3 train_wine_random_forest.py`."])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=root / "wine2.csv")
    parser.add_argument("--neural-dir", type=Path, default=root / "wine_restratified_visuals" / "wine_neural_network")
    parser.add_argument("--output", type=Path, default=root / "wine_restratified_visuals" / "wine_random_forest")
    parser.add_argument("--trees", type=int, default=DEFAULT_TREES)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.trees < 1 or args.jobs == 0:
        parser.error("trees must be positive and jobs must be nonzero")
    run(args)
    parent = args.neural_dir.parent
    if (args.output.resolve() == (parent / "wine_random_forest").resolve()
            and (parent / "wine_knn" / "metrics.json").exists()):
        shared.write_run_summary(parent)


if __name__ == "__main__":
    main()
