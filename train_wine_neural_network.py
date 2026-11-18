"""Train, validate, and test a small neural network for wine quality.

Run: python3 train_wine_neural_network.py
Dependencies: python3 -m pip install -r requirements-neural-network.txt

The data is split into approximately 70% training, 20% validation, and 10% test.
The target is the numerical quality score. quality_class is excluded because
it is derived from that target. Identical CSV rows are removed, and identical
input measurements are kept in the same split even when their scores differ.
"""

import argparse
import copy
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn



CHEMICALS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]
FEATURES = CHEMICALS + ["wine_type_red"]
SEED = 2026
SCORE_MIN, SCORE_MAX = 0, 10  # Quality scale; predictions are not clipped using test labels.


def prepare_data(csv_path):
    raw = pd.read_csv(csv_path)
    required = CHEMICALS + ["wine_type", "quality"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if raw[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if not raw.wine_type.isin(["red", "white"]).all():
        raise ValueError("wine_type must contain only red and white.")
    data = raw.drop_duplicates().copy()
    data.insert(0, "source_row", data.index)
    data = data.reset_index(drop=True)
    X = data[CHEMICALS].copy()
    X["wine_type_red"] = data.wine_type.eq("red").astype(float)
    y = data.quality.to_numpy(dtype=np.float32)
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError("Inputs and targets must be finite numbers.")
    if not ((y == np.rint(y)) & (y >= SCORE_MIN) & (y <= SCORE_MAX)).all():
        raise ValueError("Quality must be an integer on the 0–10 scale.")
    # Hash inputs only: conflicting labels for identical inputs also stay together.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    development, test = next(outer.split(X, y, groups))
    # Reserve 20% of all groups for validation, rather than 20% of the remainder.
    # An integer count also avoids floating-point rounding adding an extra group.
    validation_groups = round(0.20 * len(np.unique(groups)))
    inner = GroupShuffleSplit(n_splits=1, test_size=validation_groups, random_state=SEED + 1)
    tr, va = next(inner.split(X.iloc[development], y[development], groups[development]))
    splits = {"train": development[tr], "validation": development[va], "test": test}
    for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(groups[splits[first]]) & set(groups[splits[second]])
    assert sum(map(len, splits.values())) == len(data)
    return raw, data, X, y, splits


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


def train_candidate(hidden, inputs, targets, splits, target_mean, max_epochs, patience):
    torch.manual_seed(SEED)
    model = network(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    generator = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs["train"], targets["train"]),
        batch_size=128, shuffle=True, generator=generator,
    )
    history, best_mae, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch_x).squeeze(1), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        row = {"epoch": epoch}
        with torch.no_grad():
            for split in ["train", "validation"]:
                pred = model(inputs[split]).squeeze(1).numpy() + target_mean
                scores = metrics(targets[split].numpy() + target_mean, pred)
                row.update({f"{split}_{key}": scores[key] for key in ["mae", "rmse", "exact_accuracy", "within_one_accuracy"]})
        history.append(row)
        if row["validation_mae"] < best_mae:
            best_mae, best_epoch = row["validation_mae"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    print(f"Hidden layers {hidden}: best validation MAE={best_mae:.4f}, epoch={best_epoch}/{epoch}", flush=True)
    return model, pd.DataFrame(history), best_epoch, best_mae


def make_plots(history, best_epoch, results, predictions, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split, color in [("train", "#2166ac"), ("validation", "#b35806")]:
        axes[0].plot(history.epoch, history[f"{split}_mae"], label=split.title(), color=color)
        axes[1].plot(history.epoch, history[f"{split}_exact_accuracy"] * 100, label=split.title(), color=color)
    axes[0].set_ylabel("Mean absolute error (quality points)")
    axes[1].set_ylabel("Exact rounded-score accuracy (%)")
    for ax in axes:
        ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.6, label=f"Selected epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    fig.suptitle("Training and validation history — checkpoint selected by validation MAE")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ["Train", "Validation", "Test"]
    for ax, key, title, factor in zip(axes,
            ["mae", "exact_accuracy", "within_one_accuracy"],
            ["Mean absolute error (lower is better)", "Exact rounded-score accuracy", "Within one rounded-score point"],
            [1, 100, 100]):
        values = [results["Neural network"][split][key] * factor for split in ["train", "validation", "test"]]
        bars = ax.bar(labels, values, color=["#2166ac", "#d98f39", "#37966f"])
        ax.bar_label(bars, labels=[f"{v:.2f}" if factor == 1 else f"{v:.1f}%" for v in values], padding=4)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(values) * 1.18 if factor == 1 else 110)
    fig.suptitle("Neural network performance on separate data splits")
    fig.tight_layout()
    fig.savefig(output / "split_performance.png", dpi=170)
    plt.close(fig)

    test = predictions[predictions.split == "test"]
    actual, predicted = test.actual_quality, test.predicted_quality
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    rng = np.random.default_rng(SEED)
    axes[0].scatter(actual + rng.uniform(-0.1, 0.1, len(test)), predicted, s=12, alpha=0.35)
    low, high = min(actual.min(), predicted.min()) - 0.25, max(actual.max(), predicted.max()) + 0.25
    axes[0].plot([low, high], [low, high], "k--", label="Perfect prediction")
    axes[0].set(xlabel="Actual quality (jittered for visibility)", ylabel="Predicted quality", title="Held-out test predictions", xlim=(low, high), ylim=(low, high))
    axes[0].legend(fontsize=8)
    scores = np.arange(int(actual.min()), int(actual.max()) + 1)
    # Include any predicted scores outside the actual test range in the matrix.
    labels = np.arange(min(scores.min(), test.rounded_prediction.min()), max(scores.max(), test.rounded_prediction.max()) + 1)
    matrix = confusion_matrix(actual, test.rounded_prediction, labels=labels)
    image = axes[1].imshow(matrix, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                         color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axes[1].set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels,
                xlabel="Rounded predicted quality", ylabel="Actual quality", title="Test confusion matrix (counts)")
    axes[1].grid(False)
    fig.colorbar(image, ax=axes[1], shrink=0.7)
    axes[2].hist(predicted - actual, bins=25, color="#37966f", edgecolor="white")
    axes[2].axvline(0, color="black", linestyle="--")
    axes[2].set(xlabel="Predicted minus actual quality", ylabel="Number of wines", title="Test prediction errors")
    fig.tight_layout()
    fig.savefig(output / "test_predictions.png", dpi=170)
    plt.close(fig)

    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title, factor in zip(axes, ["mae", "exact_accuracy"],
                                    ["Test MAE (lower is better)", "Test exact rounded-score accuracy"], [1, 100]):
        values = [results[name]["test"][key] * factor for name in names]
        bars = ax.bar(names, values, color=["#37966f", "#9f9f9f", "#648db1"])
        ax.bar_label(bars, labels=[f"{v:.3f}" if factor == 1 else f"{v:.1f}%" for v in values], padding=4)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.2 if factor == 1 else 100)
    fig.tight_layout()
    fig.savefig(output / "baseline_comparison.png", dpi=170)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=root / "Combined_Wine.csv")
    parser.add_argument("--output", type=Path, default=root / "wine_neural_network")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw, data, X, y, splits = prepare_data(args.csv)
    print(f"Rows: {len(raw):,}; exact duplicate rows removed: {len(raw)-len(data):,}")
    print("Split sizes:", {key: len(idx) for key, idx in splits.items()}, flush=True)
    scaler = StandardScaler().fit(X.iloc[splits["train"]])
    inputs = {key: torch.tensor(scaler.transform(X.iloc[idx]), dtype=torch.float32) for key, idx in splits.items()}
    target_mean = float(y[splits["train"]].mean())
    targets = {key: torch.tensor(y[idx] - target_mean) for key, idx in splits.items()}

    candidates, trials = [], []
    for hidden in [(32, 16), (64, 32)]:
        model, history, epoch, val_mae = train_candidate(hidden, inputs, targets, splits, target_mean, args.epochs, args.patience)
        history.to_csv(args.output / f"history_{hidden[0]}_{hidden[1]}.csv", index=False)
        candidates.append((val_mae, hidden, model, history, epoch))
        trials.append({"hidden_layers": list(hidden), "validation_mae": val_mae, "best_epoch": epoch, "epochs_run": len(history)})
    _, hidden, model, history, best_epoch = min(candidates, key=lambda item: item[0])
    # Architecture and checkpoint are now fixed. Only now evaluate the test data.
    results = {name: {} for name in ["Neural network", "Mean baseline", "Ridge regression"]}
    predictions = []
    baselines = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge regression": Ridge(alpha=1.0),
    }
    for estimator in baselines.values():
        estimator.fit(inputs["train"].numpy(), y[splits["train"]])
    with torch.no_grad():
        for split, indices in splits.items():
            pred = model(inputs[split]).squeeze(1).numpy() + target_mean
            results["Neural network"][split] = metrics(y[indices], pred)
            for name, estimator in baselines.items():
                results[name][split] = metrics(y[indices], estimator.predict(inputs[split].numpy()))
            part = data.iloc[indices][["source_row", "wine_type", "quality"]].copy()
            part = part.rename(columns={"quality": "actual_quality"})
            part["split"], part["predicted_quality"], part["rounded_prediction"] = split, pred, rounded_scores(pred)
            predictions.append(part)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    by_score = []
    for score, part in predictions[predictions.split == "test"].groupby("actual_quality"):
        by_score.append({"quality": int(score), **metrics(part.actual_quality, part.predicted_quality)})
    pd.DataFrame(by_score).to_csv(args.output / "test_metrics_by_quality.csv", index=False)
    summary = {
        "seed": SEED, "target": "quality", "features": FEATURES,
        "raw_rows": len(raw), "duplicate_rows_removed": len(raw) - len(data),
        "split_sizes": {key: len(idx) for key, idx in splits.items()},
        "split_method": "Approximate 70/20/10 grouped random split; identical input rows stay together",
        "split_fractions": {"train": 0.70, "validation": 0.20, "test": 0.10},
        "selection_metric": "validation MAE", "selected_hidden_layers": list(hidden),
        "best_epoch": best_epoch, "candidates": trials, "results": results,
        "test_by_wine_type": {kind: metrics(part.actual_quality, part.predicted_quality)
                              for kind, part in predictions[predictions.split == "test"].groupby("wine_type")},
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "hidden_layers": hidden,
                "features": FEATURES, "target_mean": target_mean}, args.output / "model.pt")
    joblib.dump(scaler, args.output / "scaler.joblib")
    make_plots(history, best_epoch, results, predictions, args.output)
    report = [
        "# Wine quality neural network", "",
        f"Target: numerical quality score. Inputs: 11 chemical measurements plus red/white wine type.",
        f"Removed {len(raw)-len(data):,} identical rows from {len(raw):,} source rows.",
        f"70/20/10 train/validation/test split sizes: {summary['split_sizes']}. Identical inputs cannot cross splits.",
        "The scaler and target mean are fitted on training data only. quality_class is excluded to avoid target leakage.",
        f"Selected hidden layers: {hidden}; selected epoch: {best_epoch}. Selection uses validation MAE only.",
        "The test data is evaluated after architecture and checkpoint selection. Models are not refitted on validation data.", "",
        "| Model | Test MAE | Test RMSE | Test R² | Exact rounded score | Within one rounded point |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in results.items():
        s = values["test"]
        report.append(f"| {name} | {s['mae']:.3f} | {s['rmse']:.3f} | {s['r2']:.3f} | {s['exact_accuracy']:.1%} | {s['within_one_accuracy']:.1%} |")
    report.extend(["", "MAE and RMSE use continuous predictions. Accuracy rounds to the nearest integer (half-up), clipped to the 0–10 quality scale.",
                   "Within-one accuracy uses the rounded prediction, not a continuous-error threshold.",
                   "Scores near 5 and 6 dominate; aggregate accuracy can hide weak performance on rare scores. See test_metrics_by_quality.csv.",
                   "This is a single fixed random split; results may vary with other seeds and do not establish performance on a different wine population.",
                   "", "Reproduce: `python3 train_wine_neural_network.py`", "",
                   "Plots: learning_curves.png, split_performance.png, test_predictions.png, baseline_comparison.png.",
                   "Saved model: model.pt; preprocessing: scaler.joblib. Predictions and split assignments: predictions.csv."])
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(pd.DataFrame({name: value["test"] for name, value in results.items()}).T.to_string())
    print(f"Saved model, report, metrics, predictions, and plots to {args.output.resolve()}")


if __name__ == "__main__":
    main()
