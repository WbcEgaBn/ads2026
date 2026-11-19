"""Separate restratified wine workflow, preserving the legacy training code.

Run everything: python3 preprocessing_restratified.py
Run one model: python3 preprocessing_restratified.py --model neural-network
Outputs default to wine_restratified_visuals/. No class resampling is performed.
Holdouts and validation folds preserve quality_class proportions. Neural network
splits also keep identical inputs together. This training entry point does not
evaluate reserved tests; testing.py evaluates the saved neural network.
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn


CHEMICALS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]
FEATURES = CHEMICALS + ["wine_type_red"]
SEED = 2026
CANDIDATES = [(32, 16), (64, 32)]
METRIC_KEYS = ["accuracy", "macro_f1", "weighted_f1", "log_loss", "roc_auc_macro"]
CLASSES = ["Low", "Medium", "High"]
KNN_NEIGHBORS = 3
PROBABILITY_COLUMNS = [f"probability_{label}" for label in CLASSES]


def data_source(csv_path):
    """Record which input file produced the saved training outputs."""
    path = Path(csv_path)
    return {"csv": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def feature_matrix(data):
    """Use the same chemical inputs and wine encoding for both models."""
    X = data[CHEMICALS].copy()
    X["wine_type_red"] = data.wine_type.eq("red").astype(float)
    return X


def load_saved_holdout(csv_path, model_dir, metadata):
    """Validate saved row membership without predicting or scoring test rows."""
    if data_source(csv_path)["sha256"] != metadata["data_source"]["sha256"]:
        raise ValueError("The CSV does not match the source used to train this model.")
    assignments = pd.read_csv(Path(model_dir) / "partition_assignments.csv")
    holdout = assignments[assignments.stage == "holdout"]
    development = holdout.loc[holdout.split == "development", "source_row"].to_numpy()
    reserved = holdout.loc[holdout.split == "test_reserved", "source_row"].to_numpy()
    if not len(development) or not len(reserved):
        raise ValueError("Saved development and reserved-test memberships are required.")
    if len(np.unique(development)) != len(development) or len(np.unique(reserved)) != len(reserved):
        raise ValueError("Saved partition memberships contain duplicate source rows.")
    if set(development) & set(reserved):
        raise ValueError("Reserved test rows overlap the model's development/training rows.")
    if (len(development) != metadata["split_sizes"]["development"]
            or len(reserved) != metadata["split_sizes"]["test_reserved"]):
        raise ValueError("Saved partition sizes do not match the trained model.")
    raw = pd.read_csv(csv_path)
    source_rows = np.concatenate([development, reserved])
    if (not np.issubdtype(source_rows.dtype, np.integer)
            or (source_rows < 0).any() or (source_rows >= len(raw)).any()):
        raise ValueError("Saved source rows must be valid zero-based CSV row IDs.")
    X = feature_matrix(raw)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input features must be finite numbers.")
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if set(groups[development]) & set(groups[reserved]):
        raise ValueError("Identical inputs occur in both the development and test partitions.")
    return raw, X, development, reserved


def stratified_folds(X, classes, groups, folds, seed):
    """Use exact row stratification for singletons, grouped stratification otherwise."""
    classes, groups = np.asarray(classes), np.asarray(groups)
    if any(len(np.unique(groups[classes == label])) < folds for label in CLASSES):
        raise ValueError("Each class must span at least as many input groups as folds.")
    if len(np.unique(groups)) == len(groups):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return cv.split(X, classes)
    # Group constraints can make exact class proportions impossible. The greedy
    # unshuffled assignment prioritizes class balance for heterogeneous groups.
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=False)
    return cv.split(X, classes, groups)


def record_training(audit, stage, model, training, classes):
    """Audit actual training rows; each source index must occur exactly once."""
    training = np.asarray(training, dtype=int)
    if len(np.unique(training)) != len(training):
        raise ValueError("A training partition contains repeated source indices.")
    labels = np.asarray(classes)[training]
    if not np.isin(labels, CLASSES).all() or set(labels) != set(CLASSES):
        raise ValueError("Training partitions must contain all three quality classes.")
    if audit is not None:
        audit.append({"stage": stage, "model": model, "training_n": len(training),
                      "unique_source_rows": len(training), "repeated_source_rows": 0,
                      **{label: int(np.count_nonzero(labels == label)) for label in CLASSES}})


def save_partition_audit(partitions, classes, source_rows, output):
    """Save row membership and class proportions, including reserved row IDs only."""
    memberships, counts = [], []
    for stage, split, indices in partitions:
        indices = np.asarray(indices, dtype=int)
        assert len(indices) == len(np.unique(indices))
        memberships.extend({"stage": stage, "split": split, "source_row": int(source_rows[i])}
                           for i in indices)
        for label in CLASSES:
            count = int(np.count_nonzero(np.asarray(classes)[indices] == label))
            counts.append({"stage": stage, "split": split, "class": label,
                           "n": count, "total_n": len(indices), "proportion": count / len(indices)})
    pd.DataFrame(memberships).to_csv(output / "partition_assignments.csv", index=False)
    table = pd.DataFrame(counts)
    table.to_csv(output / "split_class_distribution.csv", index=False)
    labels = list(dict.fromkeys(table.stage + ": " + table.split))
    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.28)))
    left = np.zeros(len(labels))
    for label, color in zip(CLASSES, ["#b34b36", "#246d9c", "#347d58"]):
        values = table[table["class"] == label].proportion.to_numpy() * 100
        ax.barh(labels, values, left=left, color=color, label=label)
        left += values
    ax.set(xlabel="Share of rows (%)", xlim=(0, 100),
           title="Restratified partitions: observed class proportions (no resampling)")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)
    save_plot(fig, output, "split_class_distribution.png")


def prepare_data(csv_path):
    raw = pd.read_csv(csv_path)
    required = CHEMICALS + ["wine_type", "quality_class"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if raw[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if not raw.wine_type.isin(["red", "white"]).all():
        raise ValueError("wine_type must contain only red and white.")
    if not raw.quality_class.isin(CLASSES).all():
        raise ValueError("quality_class must contain only Low, Medium, and High.")
    # Numerical quality is unused, including in duplicate detection and splitting.
    data = raw[required].drop_duplicates().copy()
    data.insert(0, "source_row", data.index)
    data = data.reset_index(drop=True)
    X = feature_matrix(data)
    y = pd.Categorical(data.quality_class, categories=CLASSES).codes.astype(np.int64)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Inputs must be finite numbers.")
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    # Re-stratify a 10% reserved holdout, preserving identical-input groups.
    development, test = next(stratified_folds(X, data.quality_class, groups, 10, SEED))
    assert not set(groups[development]) & set(groups[test])
    assert len(development) + len(test) == len(data)
    return raw, data, X, y, groups, development, test


def network(hidden):
    layers = []
    previous = len(FEATURES)
    for width in hidden:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, len(CLASSES)))
    return nn.Sequential(*layers)


def metrics(actual, probabilities):
    """Evaluate encoded class targets; probability columns follow CLASSES."""
    actual = np.asarray(actual, dtype=int)
    # Normalize only for log loss. Rescaling rows before ROC AUC can break
    # tied forest scores and change their ranking through roundoff.
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities.argmax(axis=1)
    indicators = actual[:, None] == np.arange(len(CLASSES))
    return {
        "n": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=range(len(CLASSES)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=range(len(CLASSES)), average="weighted", zero_division=0)),
        "log_loss": float(log_loss(actual, probabilities / probabilities.sum(axis=1, keepdims=True),
                                   labels=range(len(CLASSES)))),
        "roc_auc_macro": float(roc_auc_score(indicators, probabilities, average="macro"))
        if set(actual) == set(range(len(CLASSES))) else None,
    }


def fit_network(hidden, X, y, classes, train_indices, epochs, seed, validation_indices=None,
                patience=40, audit=None, stage="training"):
    """Scale and train each original training row once per epoch."""
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X.iloc[train_indices])
    training = np.asarray(train_indices, dtype=int)
    record_training(audit, stage, candidate_name(hidden), training, classes)
    indices = {"train": train_indices}
    if validation_indices is not None:
        indices["validation"] = validation_indices
    inputs = {key: torch.tensor(scaler.transform(X.iloc[idx]), dtype=torch.float32)
              for key, idx in indices.items()}
    targets = torch.tensor(y[training], dtype=torch.long)
    training_inputs = torch.tensor(scaler.transform(X.iloc[training]), dtype=torch.float32)
    model = network(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(training_inputs, targets),
        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    history, best_f1, best_epoch = [], -float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        row = {"epoch": epoch}
        with torch.no_grad():
            for split, idx in indices.items():
                pred = torch.softmax(model(inputs[split]), dim=1).numpy()
                scores = metrics(y[idx], pred)
                row.update({f"{split}_{key}": scores[key] for key in METRIC_KEYS})
        history.append(row)
        if validation_indices is not None:
            if row["validation_macro_f1"] > best_f1:
                best_f1, best_epoch = row["validation_macro_f1"], epoch
            if epoch - best_epoch >= patience:
                break
    # Inner models only select an epoch count; each scored fold is trained afresh.
    return model, scaler, pd.DataFrame(history), best_epoch or epochs


def predict(model, scaler, X):
    """Return softmax probabilities in Low, Medium, High order."""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(scaler.transform(X), dtype=torch.float32)
        return torch.softmax(model(inputs), dim=1).numpy()


def candidate_name(hidden):
    return "Neural network " + "-".join(map(str, hidden))


def cross_validate(X, y, classes, groups, development, folds, epochs, patience, output, audit, partitions):
    rows, prediction_parts, histories, assignments = [], [], {}, []
    cv = stratified_folds(X.iloc[development], classes[development], groups[development], folds, SEED)
    for fold, (tr, va) in enumerate(cv, 1):
        train_idx, val_idx = development[tr], development[va]
        inner = stratified_folds(X.iloc[train_idx], classes[train_idx], groups[train_idx], 5, SEED + fold)
        it, iv = next(inner)
        inner_train, inner_val = train_idx[it], train_idx[iv]
        partitions.extend([(f"outer_fold_{fold}", "train", train_idx),
                           (f"outer_fold_{fold}", "validation", val_idx),
                           (f"inner_fold_{fold}", "train", inner_train),
                           (f"inner_fold_{fold}", "validation", inner_val)])
        assert not set(groups[train_idx]) & set(groups[val_idx])
        assert not set(groups[inner_train]) & set(groups[inner_val])
        assignments.extend({"row_index": int(idx), "fold": fold} for idx in val_idx)
        print(f"Fold {fold}/{folds}: train={len(train_idx)}, validation={len(val_idx)}", flush=True)
        for hidden in CANDIDATES:
            name = candidate_name(hidden)
            _, _, history, best_epoch = fit_network(
                hidden, X, y, classes, inner_train, epochs, SEED + fold, inner_val, patience,
                audit=audit, stage=f"inner_fold_{fold}",
            )
            stem = "_".join(map(str, hidden))
            history.to_csv(output / f"history_{stem}_fold_{fold}.csv", index=False)
            histories[(name, fold)] = (history, best_epoch)
            model, scaler, _, _ = fit_network(
                hidden, X, y, classes, train_idx, best_epoch, SEED + fold,
                audit=audit, stage=f"outer_fold_{fold}",
            )
            pred = predict(model, scaler, X.iloc[val_idx])
            rows.append({"model": name, "fold": fold, "train_n": len(train_idx),
                         "inner_train_n": len(inner_train), "inner_validation_n": len(inner_val),
                         "best_epoch": best_epoch, "epochs_run": len(history), **metrics(y[val_idx], pred)})
            prediction_parts.append(prediction_frame(val_idx, fold, name, pred))
            print(f"  {name}: validation macro F1={rows[-1]['macro_f1']:.4f}; epochs={best_epoch}", flush=True)
        # Baselines use the same original training rows as the neural networks.
        scaler = StandardScaler().fit(X.iloc[train_idx])
        training = train_idx.copy()
        train_X, val_X = scaler.transform(X.iloc[training]), scaler.transform(X.iloc[val_idx])
        for name, estimator in [("Majority baseline", DummyClassifier(strategy="most_frequent")),
                                ("Logistic regression", LogisticRegression(max_iter=1000, random_state=SEED))]:
            record_training(audit, f"outer_fold_{fold}", name, training, classes)
            estimator.fit(train_X, y[training])
            pred = estimator.predict_proba(val_X)[:, [list(estimator.classes_).index(i) for i in range(len(CLASSES))]]
            rows.append({"model": name, "fold": fold, "train_n": len(train_idx), **metrics(y[val_idx], pred)})
            prediction_parts.append(prediction_frame(val_idx, fold, name, pred))
    return pd.DataFrame(rows), pd.concat(prediction_parts, ignore_index=True), histories, pd.DataFrame(assignments)


def prediction_frame(indices, fold, name, probabilities):
    return pd.DataFrame({"row_index": indices, "fold": fold, "model": name,
                         "predicted_class": np.asarray(CLASSES)[probabilities.argmax(axis=1)],
                         **{column: probabilities[:, i] for i, column in enumerate(PROBABILITY_COLUMNS)}})


def summarize_cv(fold_metrics, predictions):
    results = {}
    for name, part in predictions.groupby("model", sort=False):
        scores = fold_metrics[fold_metrics.model == name]
        results[name] = {
            "pooled": metrics(pd.Categorical(part.actual_class, categories=CLASSES).codes,
                              part[PROBABILITY_COLUMNS].to_numpy()),
            "mean": {key: None if scores[key].dropna().empty else float(scores[key].mean()) for key in METRIC_KEYS},
            "std": {key: None if len(scores[key].dropna()) < 2 else float(scores[key].std(ddof=1)) for key in METRIC_KEYS},
        }
    return results


def save_plot(fig, output, filename):
    fig.tight_layout()
    fig.savefig(output / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_training_distribution(audit, output):
    table = pd.DataFrame(audit)
    stage = "final" if "final" in table.stage.values else "outer_fold_1"
    row = table[table.stage == stage].iloc[0]
    values = [row[label] for label in CLASSES]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASSES, values, color=["#b34b36", "#246d9c", "#347d58"])
    ax.bar_label(bars, labels=[f"{v:,} ({v / sum(values):.1%})" for v in values], padding=4)
    ax.set(ylabel="Unique training rows per epoch / fit", ylim=(0, max(values) * 1.2),
           title=f"Observed training classes: {row['model']} ({stage.replace('_', ' ')})")
    save_plot(fig, output, "training_class_distribution.png")


def make_plots(histories, selected, folds, fold_metrics, results, train_metrics, predictions, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(folds, 2, figsize=(12, 2.7 * folds), squeeze=False)
    for fold, pair in enumerate(axes, 1):
        history, best_epoch = histories[(selected, fold)]
        for split, color in [("train", "#2166ac"), ("validation", "#b35806")]:
            pair[0].plot(history.epoch, history[f"{split}_log_loss"], label=f"Inner {split}", color=color)
            pair[1].plot(history.epoch, history[f"{split}_macro_f1"], label=f"Inner {split}", color=color)
        pair[0].set_ylabel(f"Fold {fold}: cross-entropy")
        pair[1].set_ylabel("Macro F1")
        pair[1].set_ylim(0, 1)
        for ax in pair:
            ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.6, label=f"Epoch {best_epoch}")
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7)
    fig.suptitle(f"{selected}: inner early-stopping histories\nEpochs selected by inner validation macro F1", y=1.002)
    save_plot(fig, output, "learning_curves.png")

    keys = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("roc_auc_macro", "Macro ROC AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (key, title) in zip(axes, keys):
        values = [train_metrics[key], results[selected]["pooled"][key]]
        bars = ax.bar(["Final model\ntraining", "Out-of-fold\nvalidation"], values, color=["#2166ac", "#d98f39"])
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4)
        ax.set(title=title, ylim=(0, 1.1))
    fig.suptitle("Training and validation performance (different fitted models)")
    save_plot(fig, output, "split_performance.png")

    matrix = confusion_matrix(predictions.actual_class, predictions.predicted_class, labels=CLASSES)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, values, title in zip(axes, [matrix, normalized], ["Validation counts", "Recall by actual class"]):
        img = ax.imshow(values, cmap="Blues", vmin=0)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                label = str(matrix[i, j]) if values is matrix else f"{values[i, j]:.1%}"
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if values[i, j] > values.max() / 2 else "black")
        ax.set(xticks=range(3), xticklabels=CLASSES, yticks=range(3), yticklabels=CLASSES,
               xlabel="Predicted class", ylabel="Actual class", title=title)
        ax.grid(False)
        fig.colorbar(img, ax=ax, shrink=0.7)
    fig.suptitle("Out-of-fold Low / Medium / High predictions")
    save_plot(fig, output, "validation_predictions.png")
    pd.DataFrame(matrix, index=pd.Index(CLASSES, name="actual_class"), columns=CLASSES).to_csv(output / "confusion_matrix.csv")

    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axes, keys):
        means = [results[name]["mean"][key] for name in names]
        errors = [results[name]["std"][key] for name in names]
        bars = ax.bar(names, means, yerr=errors, capsize=4, color=["#648db1", "#2166ac", "#9f9f9f", "#37966f"])
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=8)
        ax.set(title=title, ylim=(0, 1.15))
        ax.tick_params(axis="x", labelsize=8, labelrotation=15)
    fig.suptitle(f"{folds}-fold classification validation (error bars: ±1 fold SD)")
    save_plot(fig, output, "baseline_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in names:
        scores = fold_metrics[fold_metrics.model == name].sort_values("fold")
        axes[0].plot(scores.fold, scores.macro_f1, marker="o", label=name)
        axes[1].plot(scores.fold, scores.accuracy, marker="o", label=name)
    for ax, title in zip(axes, ["Macro F1", "Accuracy"]):
        ax.set(xlabel="Validation fold", ylabel=title, xticks=range(1, folds + 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle("Classification validation performance by fold")
    save_plot(fig, output, "cross_validation.png")


def write_report(summary, output):
    selected, results = summary["selected_model"], summary["cross_validation"]
    report = [
        "# Wine quality neural network classifier", "",
        f"Source: `{summary['data_source']['csv']}` (SHA-256: `{summary['data_source']['sha256']}`).", "",
        "Target: quality_class — Low, Medium, or High. Numerical quality is unused and is not required in the input CSV.",
        "Inputs: 11 chemical measurements plus red/white wine type. Both quality and quality_class are excluded from features.",
        "The network outputs three logits, trains with cross-entropy, and uses softmax probabilities in Low/Medium/High order.",
        "The predicted class has the highest probability; no score rounding or quality-score thresholds are used.", "",
        f"Removed {summary['duplicate_rows_removed']:,} duplicate feature/label rows from {summary['raw_rows']:,} source rows.",
        f"Development rows: {summary['split_sizes']['development']:,}; reserved test rows: {summary['split_sizes']['test_reserved']:,}.",
        "This training workflow does not evaluate reserved tests. Run testing.py for saved-model test results.",
        "Source rows overlap earlier experiments.",
        f"Class-stratified {summary['folds']}-fold validation preserves identical-input groups.",
        "Each outer fold uses an inner approximately 80/20 stratified split to choose epochs by maximum validation macro F1.",
        "A fresh network refits the outer training rows before outer validation scoring. Scaling uses training rows only.",
        "Each training row is used once per epoch; no resampling or class weighting is applied, including for baselines.",
        f"Final training class counts: {summary['final_training_class_counts']}.",
        f"Selected architecture: {summary['selected_hidden_layers']} (highest mean outer-fold macro F1).",
        f"Final development fit: {summary['final_epochs']} epochs (rounded median of selected fold epochs).", "",
        "| Model | Accuracy, mean ± SD | Macro F1, mean ± SD | Macro ROC AUC, mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for name, values in results.items():
        m, sd = values["mean"], values["std"]
        report.append(f"| {name} | {m['accuracy']:.1%} ± {sd['accuracy']:.1%} | {m['macro_f1']:.4f} ± {sd['macro_f1']:.4f} | {m['roc_auc_macro']:.4f} ± {sd['roc_auc_macro']:.4f} |")
    pooled = results[selected]["pooled"]
    report.extend([
        "", f"Selected model pooled out-of-fold accuracy: **{pooled['accuracy']:.1%}**; macro F1: **{pooled['macro_f1']:.4f}**; macro ROC AUC: **{pooled['roc_auc_macro']:.4f}**.",
        f"Final model in-sample accuracy: {summary['final_training_metrics']['accuracy']:.1%}. This is a different fit from the fold models.", "",
        "Macro F1 gives each class equal weight. ROC curves use out-of-fold probabilities, with one class versus the rest.",
        "Fold SD describes variation across folds, not a confidence interval. Pooled metrics weight rows; fold means weight folds equally.",
        "Validation selects the architecture, so selected-model scores may be optimistic and are not independent test estimates.",
        "KNN predicts the same target but uses different partitions; its scores are not a paired model comparison.", "",
        f"Reproduce: `python3 preprocessing_restratified.py --model neural-network --folds {summary['folds']} --epochs {summary['max_epochs']} --patience {summary['patience']}`", "",
        "predictions.csv contains selected-model out-of-fold class predictions and three probabilities; cv_predictions.csv includes all models.",
        "validation_metrics_by_class.csv contains precision, recall, F1, and support. confusion_matrix.csv contains class counts.",
        "ROC outputs: roc_curves.png/.csv, roc_auc_by_class.csv, roc_auc_by_fold.png; metrics.json records pooled and fold summaries.",
        "Plots: learning_curves.png, split_performance.png, validation_predictions.png, baseline_comparison.png, cross_validation.png, training_class_distribution.png.",
        "model.pt stores the final classifier and its class order; scaler.joblib stores its fitted preprocessing.",
        "fold_metrics.csv, fold_assignments.csv, history_*_fold_*.csv, and final_history.csv audit validation and training.",
        "training_rows.csv, partition_assignments.csv, and split_class_distribution.csv audit training counts and partition membership.",
    ])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_neural_network(args):
    if args.epochs < 1 or args.patience < 1 or args.folds < 2:
        raise ValueError("epochs and patience must be positive; folds must be at least 2")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw, data, X, y, groups, development, test = prepare_data(args.csv)
    if args.folds > len(np.unique(groups[development])):
        raise ValueError("folds cannot exceed the number of development input groups")
    # Inner early stopping requires at least two distinct training groups per fold.
    min_train_groups = len(np.unique(groups[development])) - int(np.ceil(len(np.unique(groups[development])) / args.folds))
    if min_train_groups < 2:
        raise ValueError("not enough development groups for inner early stopping")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Rows: {len(raw):,}; duplicate rows removed: {len(raw)-len(data):,}", flush=True)
    print(f"Development: {len(development):,}; test reserved without evaluation: {len(test):,}", flush=True)
    classes = data.quality_class.to_numpy()
    training_audit = []
    partitions = [("holdout", "development", development), ("holdout", "test_reserved", test)]
    fold_metrics, predictions, histories, assignments = cross_validate(
        X, y, classes, groups, development, args.folds, args.epochs, args.patience, args.output, training_audit, partitions,
    )
    # Check complete OOF coverage and ensure reserved test rows never enter outputs.
    assert set(assignments.row_index) == set(development)
    assert not assignments.row_index.duplicated().any()
    assert not set(predictions.row_index) & set(test)
    for _, part in predictions.groupby("model"):
        assert set(part.row_index) == set(development) and not part.row_index.duplicated().any()
    predictions = predictions.merge(
        data[["source_row", "wine_type", "quality_class"]].rename_axis("row_index").reset_index(),
        on="row_index", validate="many_to_one",
    ).rename(columns={"quality_class": "actual_class"})
    predictions["split"] = "validation"
    results = summarize_cv(fold_metrics, predictions)
    selected = max((candidate_name(hidden) for hidden in CANDIDATES), key=lambda name: results[name]["mean"]["macro_f1"])
    hidden = next(hidden for hidden in CANDIDATES if candidate_name(hidden) == selected)
    epoch_counts = fold_metrics.loc[fold_metrics.model == selected, "best_epoch"]
    final_epochs = int(np.floor(epoch_counts.median() + 0.5))
    print(f"Selected {selected}; training final development model for {final_epochs} epochs", flush=True)
    model, scaler, final_history, _ = fit_network(
        hidden, X, y, classes, development, final_epochs, SEED, audit=training_audit, stage="final",
    )
    pd.DataFrame(training_audit).to_csv(args.output / "training_rows.csv", index=False)
    final_counts = {label: training_audit[-1][label] for label in CLASSES}
    print(f"Final unique training counts: {final_counts}", flush=True)
    train_pred = predict(model, scaler, X.iloc[development])
    train_metrics = metrics(y[development], train_pred)
    selected_predictions = predictions[predictions.model == selected].sort_values("row_index").copy()
    predictions.to_csv(args.output / "cv_predictions.csv", index=False)
    selected_predictions.to_csv(args.output / "predictions.csv", index=False)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    assignments = assignments.merge(data[["source_row"]].rename_axis("row_index").reset_index(), on="row_index", validate="one_to_one")
    assignments.to_csv(args.output / "fold_assignments.csv", index=False)
    final_history.to_csv(args.output / "final_history.csv", index=False)
    precision, recall, f1, support = precision_recall_fscore_support(
        selected_predictions.actual_class, selected_predictions.predicted_class,
        labels=CLASSES, zero_division=0,
    )
    pd.DataFrame({"class": CLASSES, "precision": precision, "recall": recall,
                  "f1": f1, "support": support}).to_csv(args.output / "validation_metrics_by_class.csv", index=False)
    pd.DataFrame([{"wine_type": key,
                   **metrics(pd.Categorical(part.actual_class, categories=CLASSES).codes,
                             part[PROBABILITY_COLUMNS].to_numpy())}
                  for key, part in selected_predictions.groupby("wine_type")]).to_csv(
                      args.output / "validation_metrics_by_wine_type.csv", index=False)
    summary = {
        "data_source": data_source(args.csv),
        "seed": SEED, "task": "multiclass_classification", "target": "quality_class", "classes": CLASSES,
        "features": FEATURES, "loss": "cross_entropy",
        "raw_rows": len(raw), "duplicate_rows_removed": len(raw) - len(data),
        "split_sizes": {"development": len(development), "test_reserved": len(test)},
        "split_method": "Class-stratified 10% holdout and nested validation, preserving identical-input groups",
        "test_evaluated": False, "folds": args.folds,
        "max_epochs": args.epochs, "patience": args.patience,
        "selection_metric": "mean outer-fold validation macro F1", "selected_model": selected,
        "selected_hidden_layers": list(hidden), "final_epochs": final_epochs,
        "cross_validation": results, "final_training_metrics": train_metrics,
        "training_policy": "Original training rows once per epoch / fit; no class resampling or class weighting",
        "final_training_class_counts": final_counts,
    }
    summary["roc_auc"] = save_neural_roc_outputs(selected_predictions, fold_metrics[fold_metrics.model == selected], args.output)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "hidden_layers": hidden,
                "data_source": summary["data_source"],
                "features": FEATURES, "target": "quality_class", "classes": CLASSES,
                "task": "multiclass_classification", "loss": "cross_entropy",
                "output_activation": "softmax", "probability_class_order": CLASSES,
                "epochs": final_epochs, "training_rows": len(development),
                "unique_training_rows": sum(final_counts.values()),
                "training_class_counts": final_counts}, args.output / "model.pt")
    joblib.dump(scaler, args.output / "scaler.joblib")
    make_plots(histories, selected, args.folds, fold_metrics, results, train_metrics, selected_predictions, args.output)
    save_partition_audit(partitions, classes, data.source_row.to_numpy(), args.output)
    plot_training_distribution(training_audit, args.output)
    write_report(summary, args.output)
    # Clean stale outputs only inside this new workflow's output directory.
    for filename in ["test_predictions.png", "test_metrics_by_quality.csv", "validation_metrics_by_quality.csv",
                     "history_32_16.csv", "history_64_32.csv"]:
        (args.output / filename).unlink(missing_ok=True)
    for path in args.output.glob("history_*_fold_*.csv"):
        if path.name not in {f"history_{h[0]}_{h[1]}_fold_{f}.csv" for h in CANDIDATES for f in range(1, args.folds + 1)}:
            path.unlink()
    print(pd.DataFrame({name: value["mean"] for name, value in results.items()}).T.to_string())
    print(f"Saved training and validation outputs to {args.output.resolve()}; test set not evaluated.")


def classification_roc_auc(actual, probabilities):
    """One-vs-rest ROC AUC with probability columns explicitly in CLASSES order."""
    actual = np.asarray(actual)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(actual), len(CLASSES)):
        raise ValueError("Probability columns must follow Low, Medium, High order.")
    if not np.isin(actual, CLASSES).all() or set(actual) != set(CLASSES):
        raise ValueError("ROC AUC requires all three quality classes.")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Class probabilities must be finite, within [0, 1], and sum to one.")
    indicators = actual[:, None] == np.asarray(CLASSES)
    per_class = {label: float(roc_auc_score(indicators[:, i], probabilities[:, i]))
                 for i, label in enumerate(CLASSES)}
    return {
        "per_class": per_class,
        "macro": float(np.mean(list(per_class.values()))),
        "weighted": float(np.average(list(per_class.values()), weights=indicators.sum(axis=0))),
        "micro": float(roc_auc_score(indicators.ravel(), probabilities.ravel())),
    }


def save_neural_roc_outputs(predictions, fold_metrics, output, model_name="Neural network classifier"):
    actual = predictions.actual_class.to_numpy()
    probabilities = predictions[PROBABILITY_COLUMNS].to_numpy()
    pooled = classification_roc_auc(actual, probabilities)
    curves, class_rows = [], []
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (label, color) in enumerate(zip(CLASSES, ["#b34b36", "#246d9c", "#347d58"])):
        fpr, tpr, thresholds = roc_curve(actual == label, probabilities[:, i])
        curves.append(pd.DataFrame({"curve": label, "false_positive_rate": fpr,
                                    "true_positive_rate": tpr, "threshold": thresholds}))
        ax.plot(fpr, tpr, color=color, label=f"{label} vs rest: AUC {pooled['per_class'][label]:.4f}")
        class_rows.append({"class": label, "positive_n": int(np.count_nonzero(actual == label)),
                           "negative_n": int(np.count_nonzero(actual != label)),
                           "pooled_roc_auc": pooled["per_class"][label]})
    indicators = actual[:, None] == np.asarray(CLASSES)
    fpr, tpr, thresholds = roc_curve(indicators.ravel(), probabilities.ravel())
    curves.append(pd.DataFrame({"curve": "micro", "false_positive_rate": fpr,
                                "true_positive_rate": tpr, "threshold": thresholds}))
    ax.plot(fpr, tpr, color="#6b4a88", ls=":", label=f"Micro average: AUC {pooled['micro']:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", label="Chance: AUC 0.5000")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate (recall)",
           title=f"{model_name}: out-of-fold ROC curves\nMacro AUC {pooled['macro']:.4f} | {len(actual):,} validation rows")
    ax.legend(loc="lower right")
    save_plot(fig, output, "roc_curves.png")
    pd.concat(curves, ignore_index=True).to_csv(output / "roc_curves.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "roc_auc_by_class.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fold_metrics.fold, fold_metrics.roc_auc_macro, marker="o")
    ax.set(xlabel="Validation fold", ylabel="Macro ROC AUC", ylim=(0, 1),
           xticks=fold_metrics.fold, title=f"{model_name}: ROC AUC by fold")
    save_plot(fig, output, "roc_auc_by_fold.png")
    return {"method": "one-vs-rest", "probability_class_order": CLASSES,
            "pooled_out_of_fold": pooled,
            "mean_fold_macro": float(fold_metrics.roc_auc_macro.mean()),
            "std_fold_macro": float(fold_metrics.roc_auc_macro.std(ddof=1))}


def save_knn_roc_outputs(chosen, fold_metrics, k, output):
    """Save ROC curves and AUC summaries using only out-of-fold probabilities."""
    probabilities = chosen[[f"probability_{label}" for label in CLASSES]].to_numpy()
    actual = chosen.actual_class.to_numpy()
    pooled = classification_roc_auc(actual, probabilities)
    selected_folds = fold_metrics.sort_values("fold")
    pd.DataFrame([
        {"class": label, "positive_n": int(np.count_nonzero(actual == label)),
         "negative_n": int(np.count_nonzero(actual != label)),
         "pooled_roc_auc": pooled["per_class"][label],
         "mean_fold_roc_auc": float(selected_folds[f"roc_auc_{label}"].mean()),
         "std_fold_roc_auc": float(selected_folds[f"roc_auc_{label}"].std(ddof=1))}
        for label in CLASSES
    ]).to_csv(output / "roc_auc_by_class.csv", index=False)

    colors = {"Low": "#b34b36", "Medium": "#246d9c", "High": "#347d58"}
    curves = []
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, label in enumerate(CLASSES):
        fpr, tpr, thresholds = roc_curve(actual == label, probabilities[:, i])
        curves.append(pd.DataFrame({"curve": label, "false_positive_rate": fpr,
                                    "true_positive_rate": tpr, "threshold": thresholds}))
        ax.plot(fpr, tpr, color=colors[label], lw=2,
                label=f"{label} vs rest: AUC {pooled['per_class'][label]:.4f}")
    indicators = actual[:, None] == np.asarray(CLASSES)
    fpr, tpr, thresholds = roc_curve(indicators.ravel(), probabilities.ravel())
    curves.append(pd.DataFrame({"curve": "micro", "false_positive_rate": fpr,
                                "true_positive_rate": tpr, "threshold": thresholds}))
    ax.plot(fpr, tpr, color="#6b4a88", ls=":", lw=2,
            label=f"Micro average: AUC {pooled['micro']:.4f}")
    ax.plot([0, 1], [0, 1], color="#777777", ls="--", label="Chance: AUC 0.5000")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate (recall)",
           title=f"KNN (K={k}): out-of-fold ROC curves\n"
                 f"Macro AUC {pooled['macro']:.4f} | {len(selected_folds)} folds | {len(actual):,} validation rows")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    save_plot(fig, output, "roc_curves.png")
    pd.concat(curves, ignore_index=True).to_csv(output / "roc_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label in CLASSES:
        ax.plot(selected_folds.fold, selected_folds[f"roc_auc_{label}"], marker="o",
                color=colors[label], label=f"{label} vs rest")
    ax.plot(selected_folds.fold, selected_folds.roc_auc_macro, color="#222222",
            marker="s", lw=2, label="Macro average")
    ax.axhline(0.5, color="#777777", ls="--", label="Chance")
    ax.set(xlabel="Validation fold", ylabel="ROC AUC", ylim=(0, 1.02),
           xticks=selected_folds.fold,
           title=f"KNN (K={k}): ROC AUC by validation fold")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(alpha=0.2)
    save_plot(fig, output, "roc_auc_by_fold.png")

    return {"method": "one-vs-rest", "probability_class_order": CLASSES,
            "pooled_out_of_fold": pooled,
            "mean_fold_macro": float(selected_folds.roc_auc_macro.mean()),
            "std_fold_macro": float(selected_folds.roc_auc_macro.std(ddof=1))}


def run_knn(args):
    """Stratified KNN validation using every training row exactly once per fit."""
    raw, _, _, _, _, _, _ = prepare_data(args.csv)
    # Preserve the original KNN source rows, stratified 80/20 split, and macro F1.
    X = feature_matrix(raw)
    classes = raw.quality_class.to_numpy()
    development, reserved = train_test_split(
        np.arange(len(raw)), test_size=0.2, random_state=SEED, stratify=classes,
    )
    if args.folds > min(np.count_nonzero(classes[development] == label) for label in CLASSES):
        raise ValueError("KNN folds cannot exceed the smallest development class count")
    args.output.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    k = KNN_NEIGHBORS
    audit, rows, predictions, assignments = [], [], [], []
    partitions = [("holdout", "development", development), ("holdout", "test_reserved", reserved)]
    print(f"KNN: development={len(development):,}; test reserved={len(reserved):,}; folds={args.folds}", flush=True)
    for fold, (tr, va) in enumerate(cv.split(X.iloc[development], classes[development]), 1):
        train_idx, val_idx = development[tr], development[va]
        scaler = StandardScaler().fit(X.iloc[train_idx])
        training = train_idx.copy()
        record_training(audit, f"outer_fold_{fold}", f"KNN (K={k})", training, classes)
        partitions.extend([(f"outer_fold_{fold}", "train", train_idx),
                           (f"outer_fold_{fold}", "validation", val_idx)])
        train_X = scaler.transform(X.iloc[training])
        val_X = scaler.transform(X.iloc[val_idx])
        assignments.extend({"source_row": int(idx), "fold": fold} for idx in val_idx)
        assert not set(training) & set(val_idx)
        assert not set(training) & set(reserved)
        model = KNeighborsClassifier(n_neighbors=int(k), weights="distance", metric="manhattan")
        model.fit(train_X, classes[training])
        predicted = model.predict(val_X)
        probabilities = model.predict_proba(val_X)[:, [list(model.classes_).index(label) for label in CLASSES]]
        auc_scores = classification_roc_auc(classes[val_idx], probabilities)
        score = f1_score(classes[val_idx], predicted, labels=CLASSES, average="macro", zero_division=0)
        rows.append({"k": int(k), "fold": fold, "macro_f1": float(score),
                     "roc_auc_macro": auc_scores["macro"], "roc_auc_micro": auc_scores["micro"],
                     "roc_auc_weighted": auc_scores["weighted"],
                     **{f"roc_auc_{label}": value for label, value in auc_scores["per_class"].items()},
                     "train_n": len(train_idx), "validation_n": len(val_idx)})
        predictions.append(pd.DataFrame({"source_row": val_idx, "fold": fold, "k": int(k),
                                          **{f"probability_{label}": probabilities[:, i] for i, label in enumerate(CLASSES)},
                                          "actual_class": classes[val_idx], "predicted_class": predicted}))
        print(f"KNN fold {fold}/{args.folds}: {len(training):,} unique training rows", flush=True)
    fold_metrics = pd.DataFrame(rows)
    mean_f1 = float(fold_metrics.macro_f1.mean())
    std_f1 = float(fold_metrics.macro_f1.std(ddof=1))
    chosen = pd.concat(predictions, ignore_index=True)
    assert set(chosen.source_row) == set(development) and not chosen.source_row.duplicated().any()
    assert not set(chosen.source_row) & set(reserved)
    chosen.to_csv(args.output / "predictions.csv", index=False)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    pd.DataFrame(assignments).to_csv(args.output / "fold_assignments.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output / "training_rows.csv", index=False)
    summary = {"data_source": data_source(args.csv),
               "raw_rows": len(raw),
               "seed": SEED, "target": "quality_class", "folds": args.folds,
               "split_sizes": {"development": len(development), "test_reserved": len(reserved)},
               "test_evaluated": False, "n_neighbors": k, "k_selection": "fixed by user",
               "mean_validation_macro_f1": mean_f1, "std_validation_macro_f1": std_f1,
               "training_policy": "Original training rows once per epoch / fit; no class resampling or class weighting"}
    summary["roc_auc"] = save_knn_roc_outputs(chosen, fold_metrics, k, args.output)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fold_metrics.fold, fold_metrics.macro_f1, marker="o")
    ax.axhline(mean_f1, color="#b63838", linestyle="--", label=f"Mean macro F1 = {mean_f1:.4f}")
    ax.set(xlabel="Validation fold", ylabel="Macro F1", ylim=(0, 1),
           title=f"KNN (fixed K={k}): stratified validation", xticks=fold_metrics.fold)
    ax.legend()
    save_plot(fig, args.output, "macro_f1_by_fold.png")
    for filename in ["tuning.csv", "knn_tuning.png", "roc_auc_by_k.png"]:
        (args.output / filename).unlink(missing_ok=True)
    save_partition_audit(partitions, classes, np.arange(len(raw)), args.output)
    plot_training_distribution(audit, args.output)
    (args.output / "report.md").write_text(
        "# KNN training and validation\n\n"
        f"Source: `{summary['data_source']['csv']}` (SHA-256: `{summary['data_source']['sha256']}`).\n\n"
        "Uses the original raw rows and stratified 80/20 development/test split. No test evaluation is performed.\n\n"
        "StandardScaler is fitted separately on each training fold. Every original training row is used once per fit, without class resampling. "
        "The neural network also uses original training rows. Stratified partitions preserve observed class proportions.\n\n"
        f"Fixed K: {k} (specified by the user); mean {args.folds}-fold macro F1: {mean_f1:.4f} ± {std_f1:.4f}. "
        "No search over K is performed. These are validation scores, not reserved-test results.\n\n"
        "## ROC AUC\n\n"
        "One-vs-rest ROC AUC uses held-out fold probabilities, with explicit Low/Medium/High column alignment. "
        "Every development row is scored once by a model trained on the other folds. The reserved test rows are excluded.\n\n"
        f"Pooled out-of-fold macro AUC: **{summary['roc_auc']['pooled_out_of_fold']['macro']:.4f}**; "
        f"weighted AUC: {summary['roc_auc']['pooled_out_of_fold']['weighted']:.4f}; "
        f"micro AUC: {summary['roc_auc']['pooled_out_of_fold']['micro']:.4f}.\n\n"
        f"Mean fold macro AUC: {summary['roc_auc']['mean_fold_macro']:.4f} ± {summary['roc_auc']['std_fold_macro']:.4f} SD. "
        "The pooled score combines validation rows; the fold mean gives each fold equal weight.\n\n"
        "| Class vs rest | Pooled ROC AUC |\n|---|---:|\n"
        + "".join(f"| {label} | {summary['roc_auc']['pooled_out_of_fold']['per_class'][label]:.4f} |\n" for label in CLASSES)
        + "\nMacro averages the three class AUCs equally; weighted averages use observed class counts; micro pools all row/class pairs. "
        "AUC measures ranking across thresholds, not classification accuracy. K is fixed at 3. "
        "K is not selected using these validation folds.\n\n"
        "Charts: roc_curves.png, roc_auc_by_fold.png, macro_f1_by_fold.png. "
        "Numeric outputs: roc_auc_by_class.csv, roc_curves.csv, fold_metrics.csv and metrics.json. "
        "predictions.csv includes the fixed K=3 model's out-of-fold class probabilities.\n\n"
        "See training_rows.csv and split_class_distribution.csv for exact class counts and macro_f1_by_fold.png for validation performance. "
        "KNN and neural network predict the same classes but use different partitions; these are not paired comparison scores.\n"
    )
    print(f"KNN fixed K={k}; mean validation macro F1={mean_f1:.4f}; test not evaluated.", flush=True)


def write_run_summary(output):
    """Keep the separate run's entry point, methods, and headline results together."""
    knn = json.loads((output / "wine_knn" / "metrics.json").read_text())
    neural = json.loads((output / "wine_neural_network" / "metrics.json").read_text())
    score = neural["cross_validation"][neural["selected_model"]]["mean"]
    charts = len(list(output.rglob("*.png")))
    report = [
        "# Restratified wine analysis", "",
        "Run the complete workflow from the repository root:", "",
        "```bash", "python3 preprocessing_restratified.py", "```", "",
        "This is the active entry point. model_training.py runs KNN with K=3; train_wine_neural_network.py runs the neural classifier.",
        "The legacy preprocessing.py is an inactive reference to the balanced experiment.",
        "Existing wine_visuals/, wine_knn/, and wine_neural_network/ outputs are preserved.", "",
        "Restratification preserves the observed Low/Medium/High proportions in each split.",
        "It does not force equal class totals. No training row is duplicated or discarded for class balancing,",
        "and neither model uses class weights. Each neural network epoch traverses its original training rows once.",
        "Scaling uses training rows only. Both quality and quality_class are excluded from model features.",
        "Both models predict Low, Medium, or High. Numerical quality is unused by training and is not required.", "",
        "KNN retains its stratified 80/20 holdout, 10 validation folds, Manhattan distance, distance weighting,",
        "and K fixed at 3 as specified. No search over K is performed.",
        "The neural network uses a newly stratified 10% holdout, 5 outer validation folds, and an inner",
        "approximately 80/20 stratified split for early stopping. Identical inputs stay in the same partition.",
        "Exact stratification uses StratifiedKFold when every input group is a singleton; otherwise",
        "StratifiedGroupKFold approximates the class proportions while preserving groups.",
        "Duplicate feature/label rows are removed before neural network splitting; numerical quality is ignored.", "",
        "Training entry points do not evaluate reserved tests. Run testing.py to evaluate the saved neural network.",
        "Neural network validation selects its architecture; KNN uses the specified K=3. Source rows overlap earlier experiments.",
        "Both models are classifiers with different split recipes; validation scores are not a paired comparison.", "",
        f"Source: `{knn['data_source']['csv']}`; SHA-256: `{knn['data_source']['sha256']}`.", "",
        f"Generated {charts} charts in this separate output tree:", "",
        "- [exploratory/](exploratory/): regenerated source-data charts.",
        "- [wine_knn/report.md](wine_knn/report.md): KNN K=3 validation, ROC AUC, and class distributions.",
        "- [wine_neural_network/report.md](wine_neural_network/report.md): network, baselines, learning curves, and validation.", "",
        f"KNN: K={knn['n_neighbors']}; mean validation macro F1 {knn['mean_validation_macro_f1']:.4f}; "
        f"pooled validation macro ROC AUC {knn['roc_auc']['pooled_out_of_fold']['macro']:.4f}.",
        f"Neural network: {neural['selected_hidden_layers']}; mean validation macro F1 {score['macro_f1']:.4f}; "
        f"accuracy {score['accuracy']:.1%}; final fit {neural['final_epochs']} epochs.", "",
        "Each model directory includes training_rows.csv (unique training counts), partition_assignments.csv",
        "(source-row membership), and split_class_distribution.csv/.png (class counts and proportions).",
        "Training outputs contain reserved row IDs as membership records. Neural network test results are saved separately in testing/.",
    ]
    test_path = output / "wine_neural_network" / "testing" / "test_metrics.json"
    if test_path.exists():
        tested = json.loads(test_path.read_text())
        current_model = output / "wine_neural_network" / "model.pt"
        if tested.get("model_sha256") == hashlib.sha256(current_model.read_bytes()).hexdigest():
            test_scores = tested["test_metrics"]
            report.extend(["", "[Neural network reserved-test results](wine_neural_network/testing/report.md): "
                           f"{test_scores['n']} rows; accuracy {test_scores['accuracy']:.1%}; "
                           f"macro F1 {test_scores['macro_f1']:.4f}; macro ROC AUC {tested['roc_auc']['macro']:.4f}."])
    forest_path = output / "wine_random_forest" / "metrics.json"
    if forest_path.exists():
        forest = json.loads(forest_path.read_text())
        hashes = forest.get("reference_neural_artifact_hashes", {})
        if hashes and all(hashlib.sha256((output / "wine_neural_network" / name).read_bytes()).hexdigest() == digest
                          for name, digest in hashes.items()):
            validation = forest["validation_metrics"]
            report.extend(["", "[Random forest on the same neural network splits](wine_random_forest/report.md). "
                           "Run `python3 train_wine_random_forest.py` after neural network training.",
                           f"Validation accuracy {validation['accuracy']:.1%}; macro F1 {validation['macro_f1']:.4f}; "
                           f"macro ROC AUC {validation['roc_auc_macro']:.4f}.",
                           "Training leaves test rows unscored. Evaluate separately with `python3 testing.py --model random-forest`."])
            forest_test_path = output / "wine_random_forest" / "testing" / "test_metrics.json"
            if forest_test_path.exists():
                forest_test = json.loads(forest_test_path.read_text())
                model_hash = hashlib.sha256((output / "wine_random_forest" / "model.joblib").read_bytes()).hexdigest()
                if forest_test.get("evaluator") == "testing.py" and forest_test.get("model_sha256") == model_hash:
                    tested = forest_test["test_metrics"]
                    report.append("[Random forest testing.py results](wine_random_forest/testing/report.md): "
                                  f"accuracy {tested['accuracy']:.1%}; macro F1 {tested['macro_f1']:.4f}; "
                                  f"macro ROC AUC {tested['roc_auc_macro']:.4f}.")
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main(argv=None, default_model="all"):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "knn", "neural-network"], default=default_model)
    parser.add_argument("--csv", type=Path, default=root / "wine2.csv")
    parser.add_argument("--output", type=Path, help="Output directory; for all models, parent of model directories")
    parser.add_argument("--folds", type=int, help="Default: 10 for KNN, 5 for neural network")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or (args.folds is not None and args.folds < 2):
        parser.error("epochs and patience must be positive; folds must be at least 2")
    destination = args.output or root / "wine_restratified_visuals"
    if destination.resolve() in {root.resolve(), (root / "wine_knn").resolve(),
                                 (root / "wine_neural_network").resolve(), (root / "wine_visuals").resolve()}:
        parser.error("Choose a separate output directory to preserve legacy outputs.")
    if args.model == "all":
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(root / "wine_visuals.py"), str(args.csv),
                        str(destination / "exploratory")], check=True)
    models = ["knn", "neural-network"] if args.model == "all" else [args.model]
    for name in models:
        run_args = argparse.Namespace(**vars(args))
        run_args.folds = args.folds if args.folds is not None else (10 if name == "knn" else 5)
        folder = "wine_knn" if name == "knn" else "wine_neural_network"
        run_args.output = destination / folder if args.model == "all" or args.output is None else destination
        (run_knn if name == "knn" else run_neural_network)(run_args)
    if args.model == "all" or (args.output is None and all(
            (destination / folder / "metrics.json").exists()
            for folder in ["wine_knn", "wine_neural_network"])):
        write_run_summary(destination)


def neural_main(argv=None):
    main(argv, default_model="neural-network")


def knn_main(argv=None):
    main(argv, default_model="knn")


if __name__ == "__main__":
    main()
