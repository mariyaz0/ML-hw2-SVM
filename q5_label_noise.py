"""Q5 - Label noise / misclassification (20 pts).

Two kinds of training-label corruption (test labels are never touched):
  - "shuffle": a random fraction of the training labels is selected and
    permuted among themselves (what the assignment describes). Only about
    half of the selected labels actually change, so the real error rate is
    roughly fraction/2.
  - "flip": a random fraction of the training labels is flipped, so the real
    error rate equals the fraction. This is the stricter test of
    misclassification.
A 100% shuffle is included as a control: if the corruption really takes
effect, the real error rate should be about 50% and performance should fall.
"""
import numpy as np
import matplotlib.pyplot as plt

from data import make_linear, make_circular, split, SEED
from best_model import make_best_svm
from utils import evaluate, print_scores, save_fig

SHUFFLE_LEVELS = [0.0, 0.15, 0.25, 0.50, 1.0]   # 1.0 = control
FLIP_LEVELS = [0.0, 0.15, 0.25, 0.50]
PLOT_LEVELS = [0.0, 0.15, 0.25, 0.50]           # levels drawn in the main figures
METRICS = ["accuracy", "f1", "auc"]


def make_rng(dataset_name, mode, fraction, dim=2):
    """Same seeding rule as Q6, so the dim=2 results match across Q5 and Q6."""
    return np.random.default_rng(
        SEED + dim * 1000 + int(round(fraction * 100)) * 10
        + (0 if dataset_name == "Linear" else 1)
        + (0 if mode == "shuffle" else 500_000)
    )


def shuffle_labels(y, fraction, rng):
    y_noisy = y.copy()
    n = int(round(fraction * len(y)))
    if n > 1:
        idx = rng.choice(len(y), size=n, replace=False)
        y_noisy[idx] = rng.permutation(y_noisy[idx])
    return y_noisy


def flip_labels(y, fraction, rng):
    y_noisy = y.copy()
    n = int(round(fraction * len(y)))
    if n > 0:
        idx = rng.choice(len(y), size=n, replace=False)
        y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy


corruptions = {
    "shuffle": (shuffle_labels, SHUFFLE_LEVELS),
    "flip": (flip_labels, FLIP_LEVELS),
}
datasets = {"Linear": make_linear(), "Circular": make_circular()}
results = {mode: {d: {} for d in datasets} for mode in corruptions}

for mode, (corrupt, levels) in corruptions.items():
    for dataset_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = split(X, y)
        for fraction in levels:
            rng = make_rng(dataset_name, mode, fraction)
            y_noisy = corrupt(y_train, fraction, rng)
            changed = np.mean(y_noisy != y_train)

            model = make_best_svm()
            model.fit(X_train, y_noisy)
            scores = evaluate(model, X_test, y_test)
            scores["labels_changed"] = changed
            results[mode][dataset_name][fraction] = scores

            print(f"\n{dataset_name} | {mode} {fraction:.0%} | "
                  f"labels actually changed: {changed:.2%}")
            print(f"  train acc vs noisy labels   : {model.score(X_train, y_noisy):.3f}")
            print(f"  train acc vs original labels: {model.score(X_train, y_train):.3f}")
            print_scores(f"{dataset_name}, {mode} {fraction:.0%}", scores)

# ---- Figures: one per metric, Linear and Circular drawn so both stay visible
styles = {
    "Linear": dict(marker="o", ms=10, ls="-", color="tab:blue", zorder=2),
    "Circular": dict(marker="s", ms=5, ls="--", color="tab:orange", zorder=3),
}


def plot_metric(mode, metric, levels, fname, title_noun):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in datasets:
        vals = [results[mode][name][f][metric] for f in levels]
        ax.plot([100 * f for f in levels], vals, label=name, **styles[name])
    ax.set_xlabel(f"Training labels {title_noun} (%)")
    ax.set_ylabel("AUC" if metric == "auc" else metric.capitalize())
    ax.set_xticks([100 * f for f in levels])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Effect of label {mode} on "
                 f"{'AUC' if metric == 'auc' else metric}")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, fname)


for metric in METRICS:
    plot_metric("shuffle", metric, PLOT_LEVELS, f"q5_label_noise_{metric}",
                "selected for random permutation")
plot_metric("flip", "accuracy", FLIP_LEVELS, "q5_label_flip_accuracy", "flipped")

# ---- Summary table to paste into the report --------------------------------
print("\n=== Summary (accuracy / F1 / AUC / labels actually changed) ===")
for mode, (_, levels) in corruptions.items():
    for name in datasets:
        for f in levels:
            s = results[mode][name][f]
            print(f"{mode:7s} {name:9s} {f:5.0%}  acc={s['accuracy']:.3f} "
                  f"f1={s['f1']:.3f} auc={s['auc']:.3f}  changed={s['labels_changed']:.1%}")
plt.show()
