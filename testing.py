"""Evaluate the saved neural classifier on its reserved Low/Medium/High test rows.

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

    assignments = pd.read_csv(model_dir / "partition_assignments.csv")
    holdout = assignments[assignments.stage == "holdout"]
    development = holdout.loc[holdout.split == "development", "source_row"].to_numpy()
    reserved = holdout.loc[holdout.split == "test_reserved", "source_row"].to_numpy()
    if not len(development) or not len(reserved):
        raise ValueError("Saved development and reserved-test memberships are required.")
    if len(np.unique(development)) != len(development) or len(np.unique(reserved)) != len(reserved):
        raise ValueError("Saved partition memberships contain duplicate source rows.")
    if set(development) & set(reserved):
        raise ValueError("Reserved test rows overlap the model's development/training rows.")
    if (len(development) != checkpoint["training_rows"]
            or len(development) != metadata["split_sizes"]["development"]
            or len(reserved) != metadata["split_sizes"]["test_reserved"]):
        raise ValueError("Saved partition sizes do not match the trained model.")

    raw = pd.read_csv(csv_path)
    source_rows = np.concatenate([development, reserved])
    if (not np.issubdtype(source_rows.dtype, np.integer)
            or (source_rows < 0).any() or (source_rows >= len(raw)).any()):
        raise ValueError("Saved source rows must be valid zero-based CSV row IDs.")
    X = training.feature_matrix(raw)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input features must be finite numbers.")
    # Check both row IDs and identical-input groups before any test predictions.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if set(groups[development]) & set(groups[reserved]):
        raise ValueError("Identical inputs occur in both the development and test partitions.")
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


def save_accuracy_plot(predictions, scores, baseline_accuracy, output):
    plt = training.plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    values = [scores["accuracy"] * 100, baseline_accuracy * 100]
    bars = axes[0].bar(["Neural network", "Training-majority\nbaseline"], values,
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
    fig.suptitle(f"Neural network: reserved-test results | {correct}/{len(predictions)} correct")
    training.save_plot(fig, output, "test_accuracy.png")
    pd.DataFrame(matrix, index=pd.Index(training.CLASSES, name="actual_class"),
                 columns=training.CLASSES).to_csv(output / "test_confusion_matrix.csv")


def save_roc_plot(predictions, auc, output):
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
           title=f"Neural network: reserved-test ROC curves\nMacro AUC {auc['macro']:.4f} | {len(actual):,} test rows")
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
    model, scaler, X, data, checkpoint = load_test_data(args.csv, model_dir)
    probabilities = training.predict(model, scaler, X)
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
    majority = max(checkpoint["training_class_counts"], key=checkpoint["training_class_counts"].get)
    baseline_accuracy = float(np.mean(predictions.actual_class == majority))
    summary = {"data_source": training.data_source(args.csv), "target": "quality_class",
               "classes": training.CLASSES, "split": "test", "test_evaluated": True,
               "model_refitted": False, "scaler_refitted": False,
               "hidden_layers": list(checkpoint["hidden_layers"]), "training_rows": checkpoint["training_rows"],
               "model_sha256": file_hash(model_dir / "model.pt"),
               "scaler_sha256": file_hash(model_dir / "scaler.joblib"),
               "partition_assignments_sha256": file_hash(model_dir / "partition_assignments.csv"),
               "test_metrics": scores, "roc_auc": auc,
               "majority_baseline": {"class_from_training": majority, "test_accuracy": baseline_accuracy},
               "per_class": class_metrics.to_dict(orient="records")}
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    class_metrics.to_csv(output / "test_metrics_by_class.csv", index=False)
    (output / "test_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    training.plt.style.use("seaborn-v0_8-whitegrid")
    save_accuracy_plot(predictions, scores, baseline_accuracy, output)
    save_roc_plot(predictions, auc, output)
    lines = ["# Neural network reserved-test evaluation", "",
             f"Evaluated **{len(predictions):,} reserved rows** with the saved {list(checkpoint['hidden_layers'])} classifier and scaler.",
             "The exact test membership comes from partition_assignments.csv. No model or scaler was refitted.",
             "Predictions are Low, Medium, or High. Numerical quality is unused.", "",
             f"Accuracy: **{scores['accuracy']:.1%}**; macro F1: **{scores['macro_f1']:.4f}**; macro ROC AUC: **{auc['macro']:.4f}**.",
             f"The training-majority baseline always predicts {majority}: test accuracy {baseline_accuracy:.1%}.", "",
             "| Class | Precision | Recall | F1 | Test rows |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row['class']} | {row['precision']:.1%} | {row['recall']:.1%} | {row['f1']:.4f} | {row['support']} |"
                 for row in summary["per_class"])
    lines.extend(["", "ROC curves use softmax probabilities with each class versus the rest.",
                  "This evaluates the reserved portion of this training run. These source rows overlap earlier experiments.", "",
                  "- [Test accuracy and confusion matrix](test_accuracy.png)",
                  "- [Test ROC curves](test_roc_curves.png)",
                  "- [Metrics](test_metrics.json) and [per-row predictions](test_predictions.csv)", "",
                  "Reproduce: `python3 testing.py` (defaults to the active saved neural network)."])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Test rows: {len(predictions):,}; accuracy: {scores['accuracy']:.1%}; "
          f"macro F1: {scores['macro_f1']:.4f}; macro ROC AUC: {auc['macro']:.4f}")
    print(f"Saved test outputs to {output.resolve()}")
    return summary


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=root / "wine2.csv")
    parser.add_argument("--model-dir", type=Path,
                        default=root / "wine_restratified_visuals" / "wine_neural_network")
    parser.add_argument("--output", type=Path, help="Default: testing/ inside the model directory")
    args = parser.parse_args(argv)
    args.output = args.output or args.model_dir / "testing"
    run_test(args)


if __name__ == "__main__":
    main()
