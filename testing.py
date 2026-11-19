"""Evaluate saved neural or random forest classifiers on their reserved test rows.

Run: python3 testing.py
The saved model, scaler, and partition assignments are used without refitting.
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve

import preprocessing_restratified as training


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_test_data(csv_path, model_dir):
    """Check artifact provenance and load the original reserved partition."""
    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    metadata = json.loads((model_dir / "metrics.json").read_text())
    source = training.data_source(csv_path)
    if any(source["sha256"] != item["data_source"]["sha256"] for item in [checkpoint, metadata]):
        raise ValueError("The CSV does not match the source used to train this model.")
    if (checkpoint.get("task") != "multiclass_classification"
            or checkpoint.get("target") != "quality_class"
            or checkpoint.get("classes") != training.CLASSES
            or checkpoint.get("probability_class_order") != training.CLASSES
            or checkpoint.get("features") != training.FEATURES):
        raise ValueError("Use the active Low/Medium/High classifier checkpoint with its original feature order.")

    raw, X, development, reserved = training.load_saved_holdout(csv_path, model_dir, metadata)
    if len(development) != checkpoint["training_rows"]:
        raise ValueError("Saved partition sizes do not match the trained model.")
    test_data = raw.iloc[reserved][["wine_type", "quality_class"]].copy()
    test_data.insert(0, "source_row", reserved)
    if not test_data.quality_class.isin(training.CLASSES).all():
        raise ValueError("Test labels must be Low, Medium, or High.")
    if set(test_data.quality_class) != set(training.CLASSES):
        raise ValueError("Test ROC curves require examples of all three classes.")

    scaler = joblib.load(model_dir / "scaler.joblib")
    if (list(scaler.feature_names_in_) != training.FEATURES
            or np.any(scaler.n_samples_seen_ != len(development))):
        raise ValueError("The scaler must match the model's development rows and feature order.")
    model = training.network(checkpoint["hidden_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, scaler, X.iloc[reserved], test_data, checkpoint


def load_forest_test_data(csv_path, model_dir):
    """Reload the trained forest and select only its saved reserved rows."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metrics.json").read_text())
    if (metadata.get("task") != "multiclass_classification"
            or metadata.get("target") != "quality_class"
            or metadata.get("classes") != training.CLASSES
            or metadata.get("features") != training.FEATURES):
        raise ValueError("Use the saved Low/Medium/High random forest and its original features.")
    if metadata.get("model_sha256") != file_hash(model_dir / "model.joblib"):
        raise ValueError("The saved forest does not match its training metadata.")
    raw, X, development, reserved = training.load_saved_holdout(csv_path, model_dir, metadata)
    pipeline = joblib.load(model_dir / "model.joblib")
    forest, scaler = pipeline.named_steps["classifier"], pipeline.named_steps["scaler"]
    if (not isinstance(forest, RandomForestClassifier)
            or list(pipeline.classes_) != list(range(len(training.CLASSES)))
            or list(scaler.feature_names_in_) != training.FEATURES
            or np.any(scaler.n_samples_seen_ != len(development))
            or metadata["training_rows"] != len(development)):
        raise ValueError("The forest pipeline must match the saved development rows and class encoding.")
    test_data = raw.iloc[reserved][["wine_type", "quality_class"]].copy()
    test_data.insert(0, "source_row", reserved)
    if set(test_data.quality_class) != set(training.CLASSES):
        raise ValueError("Test ROC curves require Low, Medium, and High examples.")
    return pipeline, X.iloc[reserved], test_data, metadata


def save_accuracy_plot(predictions, scores, baseline_accuracy, output, model_name="Neural network"):
    plt = training.plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    values = [scores["accuracy"] * 100, baseline_accuracy * 100]
    bars = axes[0].bar([model_name, "Training-majority\nbaseline"], values,
                       color=["#246d9c", "#999999"])
    axes[0].bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=5)
    axes[0].set(ylabel="Test accuracy (%)", ylim=(0, 110), title="Overall classification accuracy")
    matrix = confusion_matrix(predictions.actual_class, predictions.predicted_class, labels=training.CLASSES)
    img = axes[1].imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center",
                         color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axes[1].set(xticks=range(3), xticklabels=training.CLASSES, yticks=range(3),
                yticklabels=training.CLASSES, xlabel="Predicted class", ylabel="Actual class",
                title="Reserved-test confusion matrix")
    axes[1].grid(False)
    fig.colorbar(img, ax=axes[1], shrink=0.8)
    correct = int(np.count_nonzero(predictions.actual_class == predictions.predicted_class))
    fig.suptitle(f"{model_name}: reserved-test results | {correct}/{len(predictions)} correct")
    training.save_plot(fig, output, "test_accuracy.png")
    pd.DataFrame(matrix, index=pd.Index(training.CLASSES, name="actual_class"),
                 columns=training.CLASSES).to_csv(output / "test_confusion_matrix.csv")


def save_roc_plot(predictions, auc, output, model_name="Neural network"):
    actual = predictions.actual_class.to_numpy()
    probabilities = predictions[training.PROBABILITY_COLUMNS].to_numpy()
    curves = []
    fig, ax = training.plt.subplots(figsize=(9, 7))
    for i, (label, color) in enumerate(zip(training.CLASSES, ["#b34b36", "#246d9c", "#347d58"])):
        fpr, tpr, thresholds = roc_curve(actual == label, probabilities[:, i])
        curves.append(pd.DataFrame({"curve": label, "false_positive_rate": fpr,
                                    "true_positive_rate": tpr, "threshold": thresholds}))
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} vs rest: AUC {auc['per_class'][label]:.4f}")
    indicators = actual[:, None] == np.asarray(training.CLASSES)
    fpr, tpr, thresholds = roc_curve(indicators.ravel(), probabilities.ravel())
    curves.append(pd.DataFrame({"curve": "micro", "false_positive_rate": fpr,
                                "true_positive_rate": tpr, "threshold": thresholds}))
    ax.plot(fpr, tpr, color="#6b4a88", ls=":", lw=2, label=f"Micro average: AUC {auc['micro']:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", label="Chance: AUC 0.5000")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate (recall)",
           title=f"{model_name}: reserved-test ROC curves\nMacro AUC {auc['macro']:.4f} | {len(actual):,} test rows")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    training.save_plot(fig, output, "test_roc_curves.png")
    pd.concat(curves, ignore_index=True).to_csv(output / "test_roc_curves.csv", index=False)
    pd.DataFrame([{"class": label, "test_roc_auc": auc["per_class"][label]}
                  for label in training.CLASSES]).to_csv(output / "test_roc_auc_by_class.csv", index=False)


def run_test(args):
    """Evaluate once with the saved classifier; never fit or select a model."""
    model_dir, output = Path(args.model_dir), Path(args.output)
    torch.set_num_threads(1)
    is_forest = getattr(args, "model", "neural-network") == "random-forest"
    model_name = "Random forest" if is_forest else "Neural network"
    if is_forest:
        model, X, data, checkpoint = load_forest_test_data(args.csv, model_dir)
        order = [list(model.classes_).index(i) for i in range(len(training.CLASSES))]
        probabilities = model.predict_proba(X)[:, order]
        counts = checkpoint["final_training_class_counts"]
        model_filename = "model.joblib"
        description = f"{checkpoint['parameters']['n_estimators']}-tree random forest"
    else:
        model, scaler, X, data, checkpoint = load_test_data(args.csv, model_dir)
        probabilities = training.predict(model, scaler, X)
        counts = checkpoint["training_class_counts"]
        model_filename = "model.pt"
        description = f"{list(checkpoint['hidden_layers'])} neural classifier"
    actual = pd.Categorical(data.quality_class, categories=training.CLASSES).codes
    scores = training.metrics(actual, probabilities)
    auc = training.classification_roc_auc(data.quality_class.to_numpy(), probabilities)
    predictions = data.rename(columns={"quality_class": "actual_class"}).reset_index(drop=True)
    predictions["predicted_class"] = np.asarray(training.CLASSES)[probabilities.argmax(axis=1)]
    predictions["split"] = "test"
    for i, column in enumerate(training.PROBABILITY_COLUMNS):
        predictions[column] = probabilities[:, i]
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions.actual_class, predictions.predicted_class, labels=training.CLASSES, zero_division=0)
    class_metrics = pd.DataFrame({"class": training.CLASSES, "precision": precision,
                                  "recall": recall, "f1": f1, "support": support})
    majority = max(counts, key=counts.get)
    baseline_accuracy = float(np.mean(predictions.actual_class == majority))
    summary = {"model": model_name, "evaluator": "testing.py",
               "data_source": training.data_source(args.csv), "target": "quality_class",
               "classes": training.CLASSES, "split": "test", "test_evaluated": True,
               "model_refitted": False, "scaler_refitted": False,
               "training_rows": checkpoint["training_rows"],
               "model_sha256": file_hash(model_dir / model_filename),
               "scaler_sha256": file_hash(model_dir / "scaler.joblib"),
               "partition_assignments_sha256": file_hash(model_dir / "partition_assignments.csv"),
               "test_metrics": scores, "roc_auc": auc,
               "majority_baseline": {"class_from_training": majority, "test_accuracy": baseline_accuracy},
               "per_class": class_metrics.to_dict(orient="records")}
    if is_forest:
        summary["parameters"] = checkpoint["parameters"]
    else:
        summary["hidden_layers"] = list(checkpoint["hidden_layers"])
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    class_metrics.to_csv(output / "test_metrics_by_class.csv", index=False)
    (output / "test_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    training.plt.style.use("seaborn-v0_8-whitegrid")
    save_accuracy_plot(predictions, scores, baseline_accuracy, output, model_name=model_name)
    save_roc_plot(predictions, auc, output, model_name=model_name)
    lines = [f"# {model_name} reserved-test evaluation", "",
             f"Evaluated **{len(predictions):,} reserved rows** with the saved {description} and scaler.",
             "The exact test membership comes from partition_assignments.csv. No model or scaler was refitted.",
             "Predictions are Low, Medium, or High. Numerical quality is unused.", "",
             f"Accuracy: **{scores['accuracy']:.1%}**; macro F1: **{scores['macro_f1']:.4f}**; macro ROC AUC: **{auc['macro']:.4f}**.",
             f"The training-majority baseline always predicts {majority}: test accuracy {baseline_accuracy:.1%}.", "",
             "| Class | Precision | Recall | F1 | Test rows |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row['class']} | {row['precision']:.1%} | {row['recall']:.1%} | {row['f1']:.4f} | {row['support']} |"
                 for row in summary["per_class"])
    command = "python3 testing.py --model random-forest" if is_forest else "python3 testing.py"
    lines.extend(["", "ROC curves use class probabilities with each class versus the rest.",
                  "This evaluates the reserved portion of this training run. These source rows overlap earlier experiments.", "",
                  "- [Test accuracy and confusion matrix](test_accuracy.png)",
                  "- [Test ROC curves](test_roc_curves.png)",
                  "- [Metrics](test_metrics.json) and [per-row predictions](test_predictions.csv)", "",
                  f"Reproduce: `{command}`."])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Test rows: {len(predictions):,}; accuracy: {scores['accuracy']:.1%}; "
          f"macro F1: {scores['macro_f1']:.4f}; macro ROC AUC: {auc['macro']:.4f}")
    print(f"Saved test outputs to {output.resolve()}")
    return summary


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["neural-network", "random-forest"], default="neural-network")
    parser.add_argument("--csv", type=Path, default=root / "wine2.csv")
    parser.add_argument("--model-dir", type=Path, help="Default: selected model directory in wine_restratified_visuals/")
    parser.add_argument("--output", type=Path, help="Default: testing/ inside the model directory")
    args = parser.parse_args(argv)
    folder = "wine_random_forest" if args.model == "random-forest" else "wine_neural_network"
    args.model_dir = args.model_dir or root / "wine_restratified_visuals" / folder
    args.output = args.output or args.model_dir / "testing"
    run_test(args)


if __name__ == "__main__":
    main()
