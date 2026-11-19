"""Shared preprocessing, balanced KNN training, and balanced neural network training.

Run both: python3 preprocessing.py
Run one: python3 preprocessing.py --model neural-network --folds 5
Dependencies: python3 -m pip install -r requirements-neural-network.txt

KNN retains its stratified 80/20 split and 10-fold validation default. The neural
network retains its grouped 10% test partition and 5-fold validation default.
Both scale training features and resample Low/High to the Medium training count.
Neither evaluates its reserved test partition. Neural network inner splits
select epoch counts; identical inputs stay together throughout that workflow.
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch import nn


CHEMICALS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]
FEATURES = CHEMICALS + ["wine_type_red"]
SEED = 2026
SCORE_MIN, SCORE_MAX = 0, 10
CANDIDATES = [(32, 16), (64, 32)]
METRIC_KEYS = ["mae", "rmse", "r2", "exact_accuracy", "within_one_accuracy"]
CLASSES = ["Low", "Medium", "High"]


def feature_matrix(data):
    """Use the same chemical inputs and wine encoding for both models."""
    X = data[CHEMICALS].copy()
    X["wine_type_red"] = data.wine_type.eq("red").astype(float)
    return X


def balance_training_indices(indices, classes, seed=SEED):
    """Retain Medium and resample Low/High to its count, like the original KNN."""
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(classes)[indices]
    if not np.isin(labels, CLASSES).all():
        raise ValueError("Training labels must be Low, Medium, or High.")
    parts = {label: indices[labels == label] for label in CLASSES}
    if any(len(part) == 0 for part in parts.values()):
        raise ValueError("Every training partition must contain Low, Medium, and High rows.")
    count = len(parts["Medium"])
    balanced = np.concatenate([
        parts["Medium"],
        resample(parts["Low"], replace=True, n_samples=count, random_state=seed),
        resample(parts["High"], replace=True, n_samples=count, random_state=seed),
    ])
    assert set(balanced).issubset(set(indices))
    assert all(np.count_nonzero(np.asarray(classes)[balanced] == label) == count for label in CLASSES)
    return balanced


def record_balance(audit, stage, model, original, balanced, classes):
    if audit is None:
        return
    classes = np.asarray(classes)
    row = {"stage": stage, "model": model, "original_n": len(original), "balanced_n": len(balanced)}
    for label in CLASSES:
        row[f"{label}_before"] = int(np.count_nonzero(classes[original] == label))
        row[f"{label}_after"] = int(np.count_nonzero(classes[balanced] == label))
    audit.append(row)


def prepare_data(csv_path):
    raw = pd.read_csv(csv_path)
    required = CHEMICALS + ["wine_type", "quality", "quality_class"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if raw[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if not raw.wine_type.isin(["red", "white"]).all():
        raise ValueError("wine_type must contain only red and white.")
    if not raw.quality_class.isin(CLASSES).all():
        raise ValueError("quality_class must contain only Low, Medium, and High.")
    data = raw.drop_duplicates().copy()
    data.insert(0, "source_row", data.index)
    data = data.reset_index(drop=True)
    X = feature_matrix(data)
    y = data.quality.to_numpy(dtype=np.float32)
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError("Inputs and targets must be finite numbers.")
    if not ((y == np.rint(y)) & (y >= SCORE_MIN) & (y <= SCORE_MAX)).all():
        raise ValueError("Quality must be an integer on the 0–10 scale.")
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    # Preserve the original test assignment; no test tensors or predictions are made.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    development, test = next(outer.split(X, y, groups))
    assert not set(groups[development]) & set(groups[test])
    assert len(development) + len(test) == len(data)
    return raw, data, X, y, groups, development, test


def network(hidden):
    layers = []
    previous = len(FEATURES)
    for width in hidden:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def rounded_scores(predictions):
    return np.clip(np.floor(np.asarray(predictions) + 0.5), SCORE_MIN, SCORE_MAX).astype(int)


def metrics(actual, predicted):
    rounded = rounded_scores(predicted)
    return {
        "n": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None,
        "exact_accuracy": float(np.mean(rounded == actual)),
        "within_one_accuracy": float(np.mean(np.abs(rounded - actual) <= 1)),
    }


def fit_network(hidden, X, y, classes, train_indices, epochs, seed, validation_indices=None,
                patience=40, audit=None, stage="training"):
    """Scale original training rows, then train on equal Low/Medium/High counts."""
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X.iloc[train_indices])
    balanced = balance_training_indices(train_indices, classes)
    record_balance(audit, stage, candidate_name(hidden), train_indices, balanced, classes)
    target_mean = float(y[balanced].mean())
    indices = {"train": train_indices}
    if validation_indices is not None:
        indices["validation"] = validation_indices
    inputs = {key: torch.tensor(scaler.transform(X.iloc[idx]), dtype=torch.float32)
              for key, idx in indices.items()}
    targets = torch.tensor(y[balanced] - target_mean)
    training_inputs = torch.tensor(scaler.transform(X.iloc[balanced]), dtype=torch.float32)
    model = network(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(training_inputs, targets),
        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    history, best_mae, best_epoch = [], float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch_x).squeeze(1), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        row = {"epoch": epoch}
        with torch.no_grad():
            for split, idx in indices.items():
                pred = model(inputs[split]).squeeze(1).numpy() + target_mean
                scores = metrics(y[idx], pred)
                row.update({f"{split}_{key}": scores[key] for key in METRIC_KEYS})
        history.append(row)
        if validation_indices is not None:
            if row["validation_mae"] < best_mae:
                best_mae, best_epoch = row["validation_mae"], epoch
            if epoch - best_epoch >= patience:
                break
    # Inner models only select an epoch count; each scored fold is trained afresh.
    return model, scaler, target_mean, pd.DataFrame(history), best_epoch or epochs


def predict(model, scaler, target_mean, X):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(scaler.transform(X), dtype=torch.float32)
        return model(inputs).squeeze(1).numpy() + target_mean


def candidate_name(hidden):
    return "Neural network " + "-".join(map(str, hidden))


def cross_validate(X, y, classes, groups, development, folds, epochs, patience, output, audit):
    rows, prediction_parts, histories, assignments = [], [], {}, []
    cv = GroupKFold(n_splits=folds)
    for fold, (tr, va) in enumerate(cv.split(X.iloc[development], groups=groups[development]), 1):
        train_idx, val_idx = development[tr], development[va]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + fold)
        it, iv = next(inner.split(X.iloc[train_idx], groups=groups[train_idx]))
        inner_train, inner_val = train_idx[it], train_idx[iv]
        assert not set(groups[train_idx]) & set(groups[val_idx])
        assert not set(groups[inner_train]) & set(groups[inner_val])
        assignments.extend({"row_index": int(idx), "fold": fold} for idx in val_idx)
        print(f"Fold {fold}/{folds}: train={len(train_idx)}, validation={len(val_idx)}", flush=True)
        for hidden in CANDIDATES:
            name = candidate_name(hidden)
            _, _, _, history, best_epoch = fit_network(
                hidden, X, y, classes, inner_train, epochs, SEED + fold, inner_val, patience,
                audit=audit, stage=f"inner_fold_{fold}",
            )
            stem = "_".join(map(str, hidden))
            history.to_csv(output / f"history_{stem}_fold_{fold}.csv", index=False)
            histories[(name, fold)] = (history, best_epoch)
            model, scaler, target_mean, _, _ = fit_network(
                hidden, X, y, classes, train_idx, best_epoch, SEED + fold,
                audit=audit, stage=f"outer_fold_{fold}",
            )
            pred = predict(model, scaler, target_mean, X.iloc[val_idx])
            rows.append({"model": name, "fold": fold, "train_n": len(train_idx),
                         "inner_train_n": len(inner_train), "inner_validation_n": len(inner_val),
                         "best_epoch": best_epoch, "epochs_run": len(history), **metrics(y[val_idx], pred)})
            prediction_parts.append(pd.DataFrame({"row_index": val_idx, "fold": fold, "model": name,
                                                  "predicted_quality": pred}))
            print(f"  {name}: validation MAE={rows[-1]['mae']:.4f}; epochs={best_epoch}", flush=True)
        # All models use the same training-only balancing policy.
        scaler = StandardScaler().fit(X.iloc[train_idx])
        balanced = balance_training_indices(train_idx, classes)
        train_X, val_X = scaler.transform(X.iloc[balanced]), scaler.transform(X.iloc[val_idx])
        for name, estimator in [("Mean baseline", DummyRegressor()), ("Ridge regression", Ridge(alpha=1.0))]:
            record_balance(audit, f"outer_fold_{fold}", name, train_idx, balanced, classes)
            estimator.fit(train_X, y[balanced])
            pred = estimator.predict(val_X)
            rows.append({"model": name, "fold": fold, "train_n": len(train_idx), **metrics(y[val_idx], pred)})
            prediction_parts.append(pd.DataFrame({"row_index": val_idx, "fold": fold, "model": name,
                                                  "predicted_quality": pred}))
    return pd.DataFrame(rows), pd.concat(prediction_parts, ignore_index=True), histories, pd.DataFrame(assignments)


def summarize_cv(fold_metrics, predictions):
    results = {}
    for name, part in predictions.groupby("model", sort=False):
        scores = fold_metrics[fold_metrics.model == name]
        results[name] = {
            "pooled": metrics(part.actual_quality, part.predicted_quality),
            "mean": {key: None if scores[key].dropna().empty else float(scores[key].mean()) for key in METRIC_KEYS},
            "std": {key: None if len(scores[key].dropna()) < 2 else float(scores[key].std(ddof=1)) for key in METRIC_KEYS},
        }
    return results


def save_plot(fig, output, filename):
    fig.tight_layout()
    fig.savefig(output / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_training_balance(audit, output):
    table = pd.DataFrame(audit)
    stage = "final" if "final" in table.stage.values else "outer_fold_1"
    row = table[table.stage == stage].iloc[0]
    before = [row[f"{label}_before"] for label in CLASSES]
    after = [row[f"{label}_after"] for label in CLASSES]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASSES))
    for offset, values, color, label in [(-0.19, before, "#648db1", "Before resampling"),
                                        (0.19, after, "#d98f39", "After resampling")]:
        bars = ax.bar(x + offset, values, width=0.38, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=4)
    ax.set(xticks=x, xticklabels=CLASSES, ylabel="Training samples per epoch / fit",
           title=f"Training class balance: {row['model']} ({stage.replace('_', ' ')})")
    ax.set_ylim(0, max(after + before) * 1.2)
    ax.legend(loc="upper center", ncol=2)
    save_plot(fig, output, "training_class_balance.png")


def make_plots(histories, selected, folds, fold_metrics, results, train_metrics, predictions, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(folds, 2, figsize=(12, 2.7 * folds), squeeze=False)
    for fold, pair in enumerate(axes, 1):
        history, best_epoch = histories[(selected, fold)]
        for split, color in [("train", "#2166ac"), ("validation", "#b35806")]:
            pair[0].plot(history.epoch, history[f"{split}_mae"], label=f"Inner {split}", color=color)
            pair[1].plot(history.epoch, history[f"{split}_exact_accuracy"] * 100, label=f"Inner {split}", color=color)
        pair[0].set_ylabel(f"Fold {fold}: MAE")
        pair[1].set_ylabel("Exact rounded accuracy (%)")
        for ax in pair:
            ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.6, label=f"Epoch {best_epoch}")
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7)
    fig.suptitle(f"{selected}: inner early-stopping histories\nOuter validation folds are excluded from epoch selection", y=1.002)
    save_plot(fig, output, "learning_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, title, factor in zip(axes, ["mae", "exact_accuracy", "within_one_accuracy"],
            ["Mean absolute error (quality points)", "Exact rounded-score accuracy (%)", "Within one rounded point (%)"], [1, 100, 100]):
        values = [train_metrics[key] * factor, results[selected]["pooled"][key] * factor]
        bars = ax.bar(["Final model\ntraining", "Out-of-fold\nvalidation"], values, color=["#2166ac", "#d98f39"])
        ax.bar_label(bars, labels=[f"{v:.3f}" if factor == 1 else f"{v:.1f}%" for v in values], padding=4)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(values) * 1.2 if factor == 1 else 110)
    fig.suptitle("Training fit and cross-validation performance (different fitted models)")
    save_plot(fig, output, "split_performance.png")

    actual, predicted = predictions.actual_quality, predictions.predicted_quality
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    rng = np.random.default_rng(SEED)
    axes[0].scatter(actual + rng.uniform(-0.1, 0.1, len(predictions)), predicted, s=12, alpha=0.25)
    low, high = min(actual.min(), predicted.min()) - 0.25, max(actual.max(), predicted.max()) + 0.25
    axes[0].plot([low, high], [low, high], "k--", label="Perfect prediction")
    axes[0].set(xlabel="Actual quality (jittered)", ylabel="Predicted quality", title="Out-of-fold validation predictions", xlim=(low, high), ylim=(low, high))
    axes[0].legend(fontsize=8)
    labels = np.arange(min(actual.min(), predictions.rounded_prediction.min()), max(actual.max(), predictions.rounded_prediction.max()) + 1).astype(int)
    matrix = confusion_matrix(actual, predictions.rounded_prediction, labels=labels)
    image = axes[1].imshow(matrix, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                         color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axes[1].set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels,
                xlabel="Rounded predicted quality", ylabel="Actual quality", title="Validation confusion matrix (counts)")
    axes[1].grid(False)
    fig.colorbar(image, ax=axes[1], shrink=0.7)
    axes[2].hist(predicted - actual, bins=30, color="#d98f39", edgecolor="white")
    axes[2].axvline(0, color="black", linestyle="--")
    axes[2].set(xlabel="Predicted minus actual quality", ylabel="Number of wines", title="Out-of-fold validation errors")
    save_plot(fig, output, "validation_predictions.png")

    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title, factor in zip(axes, ["mae", "exact_accuracy"],
            ["Mean validation MAE (lower is better)", "Mean validation exact accuracy (%)"], [1, 100]):
        means = [results[name]["mean"][key] * factor for name in names]
        errors = [results[name]["std"][key] * factor for name in names]
        bars = ax.bar(names, means, yerr=errors, capsize=5, color=["#648db1", "#2166ac", "#9f9f9f", "#37966f"])
        ax.bar_label(bars, labels=[f"{v:.3f}" if factor == 1 else f"{v:.1f}%" for v in means], padding=10)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8, labelrotation=12)
        ax.set_ylim(0, max(np.array(means) + errors) * 1.2)
    fig.suptitle(f"{folds}-fold validation comparison (error bars: ±1 fold standard deviation)")
    save_plot(fig, output, "baseline_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in names:
        scores = fold_metrics[fold_metrics.model == name].sort_values("fold")
        axes[0].plot(scores.fold, scores.mae, marker="o", label=name)
        axes[1].plot(scores.fold, scores.exact_accuracy * 100, marker="o", label=name)
    for ax in axes:
        ax.set_xlabel("Validation fold")
        ax.set_xticks(range(1, folds + 1))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("MAE (quality points)")
    axes[1].set_ylabel("Exact rounded-score accuracy (%)")
    fig.suptitle("Validation performance by fold")
    save_plot(fig, output, "cross_validation.png")


def write_report(summary, output):
    selected, results = summary["selected_model"], summary["cross_validation"]
    report = [
        "# Wine quality neural network: training and validation", "",
        "Target: numerical quality score. Inputs: 11 chemical measurements plus red/white wine type.",
        f"Removed {summary['duplicate_rows_removed']:,} identical rows from {summary['raw_rows']:,} source rows.",
        f"Development rows: {summary['split_sizes']['development']:,}; reserved test rows: {summary['split_sizes']['test_reserved']:,}.",
        "**No test-set prediction or evaluation is performed by this workflow.** The original test assignment is preserved.",
        "Earlier repository outputs evaluated this partition; it is not a historically unseen test set.", "",
        f"Grouped {summary['folds']}-fold cross-validation uses only development data. Identical inputs stay together.",
        "Each fold uses an inner grouped 80/20 split to select an epoch count by early stopping on inner validation MAE.",
        "A fresh network then trains on the full outer training fold for that epoch count before outer validation scoring.",
        "Scalers and target means are fitted separately using only the corresponding training rows. quality_class is excluded.",
        "For every inner fit, outer refit, and final fit, Low and High rows are sampled with replacement to match the Medium training count.",
        "The same balancing helper is used by KNN and the neural network. Baselines also use balanced training rows.",
        "Scaling fits the original training rows before resampling; target centering uses the balanced training targets.",
        "Training metrics describe the original training rows; validation folds retain their original class proportions.",
        f"Final training counts after resampling: {summary['final_training_class_counts']}.",
        f"Selected architecture: {summary['selected_hidden_layers']} (lowest mean fold MAE).",
        f"Final model trained on all development rows for {summary['final_epochs']} epochs (rounded median of selected fold epochs).", "",
        "| Model | Validation MAE, mean ± SD | RMSE, mean ± SD | Exact accuracy, mean ± SD | Within one point, mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in results.items():
        m, s = values["mean"], values["std"]
        report.append(f"| {name} | {m['mae']:.3f} ± {s['mae']:.3f} | {m['rmse']:.3f} ± {s['rmse']:.3f} | {m['exact_accuracy']:.1%} ± {s['exact_accuracy']:.1%} | {m['within_one_accuracy']:.1%} ± {s['within_one_accuracy']:.1%} |")
    pooled = results[selected]["pooled"]
    train = summary["final_training_metrics"]
    report.extend([
        "", f"Selected model pooled out-of-fold MAE: {pooled['mae']:.3f}; RMSE: {pooled['rmse']:.3f}; exact accuracy: {pooled['exact_accuracy']:.1%}; within one point: {pooled['within_one_accuracy']:.1%}.",
        f"Final model in-sample training MAE: {train['mae']:.3f}; exact accuracy: {train['exact_accuracy']:.1%}. This is a different fit from the fold models.", "",
        "MAE and RMSE use continuous predictions. Accuracy uses half-up rounding clipped to the 0–10 quality scale.",
        "Fold SD describes variation across folds, not a confidence interval. Pooled metrics weight rows; fold means weight folds equally.",
        "Cross-validation also selects the architecture, so selected-model scores may be optimistic and are not independent test estimates.",
        "Rare quality scores are underrepresented. See validation_metrics_by_quality.csv and validation_metrics_by_wine_type.csv.", "",
        f"Reproduce: `python3 train_wine_neural_network.py --folds {summary['folds']} --epochs {summary['max_epochs']} --patience {summary['patience']}`", "",
        "Plots: learning_curves.png, split_performance.png, validation_predictions.png, baseline_comparison.png, cross_validation.png, training_class_balance.png.",
        "Saved final model: model.pt; preprocessing: scaler.joblib. predictions.csv contains selected-model out-of-fold predictions only.",
        "cv_predictions.csv contains all candidates and baselines. fold_metrics.csv and fold_assignments.csv provide the validation audit trail.",
        "final_history.csv describes the final full-development fit; history_*_fold_*.csv files describe inner early stopping.",
        "training_balance.csv records class counts before and after resampling for every model fit.",
        "Training implementations and the shared resampling helper are in preprocessing.py.",
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
    balance_audit = []
    fold_metrics, predictions, histories, assignments = cross_validate(
        X, y, classes, groups, development, args.folds, args.epochs, args.patience, args.output, balance_audit,
    )
    # Check complete OOF coverage and ensure reserved test rows never enter outputs.
    assert set(assignments.row_index) == set(development)
    assert not assignments.row_index.duplicated().any()
    assert not set(predictions.row_index) & set(test)
    for _, part in predictions.groupby("model"):
        assert set(part.row_index) == set(development) and not part.row_index.duplicated().any()
    predictions = predictions.merge(
        data[["source_row", "wine_type", "quality"]].rename_axis("row_index").reset_index(),
        on="row_index", validate="many_to_one",
    ).rename(columns={"quality": "actual_quality"})
    predictions["rounded_prediction"] = rounded_scores(predictions.predicted_quality)
    predictions["split"] = "validation"
    results = summarize_cv(fold_metrics, predictions)
    selected = min((candidate_name(hidden) for hidden in CANDIDATES), key=lambda name: results[name]["mean"]["mae"])
    hidden = next(hidden for hidden in CANDIDATES if candidate_name(hidden) == selected)
    epoch_counts = fold_metrics.loc[fold_metrics.model == selected, "best_epoch"]
    final_epochs = int(np.floor(epoch_counts.median() + 0.5))
    print(f"Selected {selected}; training final development model for {final_epochs} epochs", flush=True)
    model, scaler, target_mean, final_history, _ = fit_network(
        hidden, X, y, classes, development, final_epochs, SEED, audit=balance_audit, stage="final",
    )
    pd.DataFrame(balance_audit).to_csv(args.output / "training_balance.csv", index=False)
    final_counts = {label: balance_audit[-1][f"{label}_after"] for label in CLASSES}
    print(f"Final balanced training counts: {final_counts}", flush=True)
    train_pred = predict(model, scaler, target_mean, X.iloc[development])
    train_metrics = metrics(y[development], train_pred)
    selected_predictions = predictions[predictions.model == selected].sort_values("row_index").copy()
    predictions.to_csv(args.output / "cv_predictions.csv", index=False)
    selected_predictions.to_csv(args.output / "predictions.csv", index=False)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    assignments = assignments.merge(data[["source_row"]].rename_axis("row_index").reset_index(), on="row_index", validate="one_to_one")
    assignments.to_csv(args.output / "fold_assignments.csv", index=False)
    final_history.to_csv(args.output / "final_history.csv", index=False)
    for column, filename in [("actual_quality", "validation_metrics_by_quality.csv"), ("wine_type", "validation_metrics_by_wine_type.csv")]:
        pd.DataFrame([{column: key, **metrics(part.actual_quality, part.predicted_quality)}
                      for key, part in selected_predictions.groupby(column)]).to_csv(args.output / filename, index=False)
    summary = {
        "seed": SEED, "target": "quality", "features": FEATURES,
        "raw_rows": len(raw), "duplicate_rows_removed": len(raw) - len(data),
        "split_sizes": {"development": len(development), "test_reserved": len(test)},
        "split_method": "Original grouped 10% holdout reserved; GroupKFold on development only",
        "test_evaluated": False, "folds": args.folds,
        "max_epochs": args.epochs, "patience": args.patience,
        "selection_metric": "mean outer-fold validation MAE", "selected_model": selected,
        "selected_hidden_layers": list(hidden), "final_epochs": final_epochs,
        "cross_validation": results, "final_training_metrics": train_metrics,
        "training_balance": "Keep Medium; sample Low and High with replacement to match Medium; training partitions only",
        "final_training_class_counts": final_counts,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "hidden_layers": hidden,
                "features": FEATURES, "target_mean": target_mean,
                "epochs": final_epochs, "training_rows": len(development),
                "balanced_training_rows": sum(final_counts.values()),
                "training_class_counts": final_counts}, args.output / "model.pt")
    joblib.dump(scaler, args.output / "scaler.joblib")
    make_plots(histories, selected, args.folds, fold_metrics, results, train_metrics, selected_predictions, args.output)
    plot_training_balance(balance_audit, args.output)
    write_report(summary, args.output)
    # Remove only obsolete generated outputs from the old single-split workflow.
    for filename in ["test_predictions.png", "test_metrics_by_quality.csv", "history_32_16.csv", "history_64_32.csv"]:
        (args.output / filename).unlink(missing_ok=True)
    for path in args.output.glob("history_*_fold_*.csv"):
        if path.name not in {f"history_{h[0]}_{h[1]}_fold_{f}.csv" for h in CANDIDATES for f in range(1, args.folds + 1)}:
            path.unlink()
    print(pd.DataFrame({name: value["mean"] for name, value in results.items()}).T.to_string())
    print(f"Saved training and validation outputs to {args.output.resolve()}; test set not evaluated.")


def run_knn(args):
    """The original KNN recipe, with scaling and resampling inside each fold."""
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
    k_values = np.arange(1, 22, 2)
    audit, rows, predictions, assignments = [], [], [], []
    print(f"KNN: development={len(development):,}; test reserved={len(reserved):,}; folds={args.folds}", flush=True)
    for fold, (tr, va) in enumerate(cv.split(X.iloc[development], classes[development]), 1):
        train_idx, val_idx = development[tr], development[va]
        scaler = StandardScaler().fit(X.iloc[train_idx])
        balanced = balance_training_indices(train_idx, classes)
        record_balance(audit, f"outer_fold_{fold}", "KNN (all K values)", train_idx, balanced, classes)
        train_X = scaler.transform(X.iloc[balanced])
        val_X = scaler.transform(X.iloc[val_idx])
        assignments.extend({"source_row": int(idx), "fold": fold} for idx in val_idx)
        assert not set(balanced) & set(val_idx)
        assert not set(balanced) & set(reserved)
        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=int(k), weights="distance", metric="manhattan")
            model.fit(train_X, classes[balanced])
            predicted = model.predict(val_X)
            score = f1_score(classes[val_idx], predicted, labels=CLASSES, average="macro", zero_division=0)
            rows.append({"k": int(k), "fold": fold, "macro_f1": float(score), "train_n": len(train_idx), "validation_n": len(val_idx)})
            predictions.append(pd.DataFrame({"source_row": val_idx, "fold": fold, "k": int(k),
                                              "actual_class": classes[val_idx], "predicted_class": predicted}))
        print(f"KNN fold {fold}/{args.folds}: {len(balanced) // 3:,} training samples per class", flush=True)
    fold_metrics = pd.DataFrame(rows)
    scores = fold_metrics.groupby("k").macro_f1.agg(["mean", "std"]).reset_index()
    best = scores.loc[scores["mean"].idxmax()]
    best_k = int(best.k)
    all_predictions = pd.concat(predictions, ignore_index=True)
    chosen = all_predictions[all_predictions.k == best_k]
    assert set(chosen.source_row) == set(development) and not chosen.source_row.duplicated().any()
    assert not set(chosen.source_row) & set(reserved)
    chosen.to_csv(args.output / "predictions.csv", index=False)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    scores.to_csv(args.output / "tuning.csv", index=False)
    pd.DataFrame(assignments).to_csv(args.output / "fold_assignments.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output / "training_balance.csv", index=False)
    summary = {"seed": SEED, "target": "quality_class", "folds": args.folds,
               "split_sizes": {"development": len(development), "test_reserved": len(reserved)},
               "test_evaluated": False, "best_k": best_k, "mean_validation_macro_f1": float(best["mean"]),
               "std_validation_macro_f1": float(best["std"]),
               "training_balance": "Keep Medium; sample Low and High with replacement to match Medium; training partitions only"}
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(scores.k, scores["mean"], yerr=scores["std"], marker="o", capsize=4)
    ax.axvline(best_k, color="#b63838", linestyle="--", label=f"Best K = {best_k}")
    ax.set(xlabel="Number of neighbors (K)", ylabel="Validation macro F1",
           title=f"KNN: {args.folds}-fold validation with balanced training folds", xticks=k_values)
    ax.legend()
    save_plot(fig, args.output, "knn_tuning.png")
    plot_training_balance(audit, args.output)
    (args.output / "report.md").write_text(
        "# KNN training and validation\n\n"
        "Uses the original raw rows and stratified 80/20 development/test split. No test evaluation is performed.\n\n"
        "StandardScaler is fitted separately on each training fold. Medium training rows are retained; Low and High are resampled with replacement to match Medium. "
        "The neural network uses the same balancing helper in preprocessing.py. Validation rows retain their original distribution.\n\n"
        f"Selected K: {best_k}; mean {args.folds}-fold macro F1: {best['mean']:.4f} ± {best['std']:.4f}. "
        "This score also selects K and is not an independent test estimate.\n\n"
        "See training_balance.csv for exact class counts and knn_tuning.png for the tuning curve. "
        "KNN and neural network scores use different targets and split recipes and should not be compared directly.\n"
    )
    print(f"KNN best K={best_k}; mean validation macro F1={best['mean']:.4f}; test not evaluated.", flush=True)


def main(argv=None, default_model="all"):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "knn", "neural-network"], default=default_model)
    parser.add_argument("--csv", type=Path, default=root / "Combined_Wine.csv")
    parser.add_argument("--output", type=Path, help="Output directory; for all models, parent of model directories")
    parser.add_argument("--folds", type=int, help="Default: 10 for KNN, 5 for neural network")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or (args.folds is not None and args.folds < 2):
        parser.error("epochs and patience must be positive; folds must be at least 2")
    models = ["knn", "neural-network"] if args.model == "all" else [args.model]
    for name in models:
        run_args = argparse.Namespace(**vars(args))
        run_args.folds = args.folds if args.folds is not None else (10 if name == "knn" else 5)
        folder = "wine_knn" if name == "knn" else "wine_neural_network"
        run_args.output = (args.output or root) / folder if args.model == "all" else (args.output or root / folder)
        (run_knn if name == "knn" else run_neural_network)(run_args)


def neural_main(argv=None):
    main(argv, default_model="neural-network")


def knn_main(argv=None):
    main(argv, default_model="knn")


if __name__ == "__main__":
    main()
