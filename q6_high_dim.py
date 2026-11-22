"""Q6 - High dimensionality (10 pts).

Repeats the Q5 shuffling analysis for dim = 2, 5, 10, 15, and prints a
geometry diagnostic (distance of each class from the origin) so the report
can explain WHY performance changes with dimension.
"""
import numpy as np
import matplotlib.pyplot as plt

from data import make_linear, make_circular, split, SEED
from best_model import make_best_svm
from utils import evaluate, print_scores, save_fig

DIMS = [2, 5, 10, 15]
NOISE_LEVELS = [0.0, 0.15, 0.25, 0.50]
METRICS = ["accuracy", "f1", "auc"]


def make_rng(dataset_name, mode, fraction, dim=2):
    """Identical to the rule in q5_label_noise.py (so dim=2 matches Q5)."""
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


generators = {"Linear": make_linear, "Circular": make_circular}
results = {name: {f: [] for f in NOISE_LEVELS} for name in generators}

for dataset_name, generator in generators.items():
    for dim in DIMS:
        X, y = generator(dim=dim)
        X_train, X_test, y_train, y_test = split(X, y)

        # Geometry diagnostic: how far each class sits from the origin.
        r = np.linalg.norm(X, axis=1)
        print(f"\n[{dataset_name}, dim={dim}] mean distance from origin: "
              f"class0={r[y == 0].mean():.2f} (sd {r[y == 0].std():.2f}), "
              f"class1={r[y == 1].mean():.2f} (sd {r[y == 1].std():.2f})")

        for fraction in NOISE_LEVELS:
            rng = make_rng(dataset_name, "shuffle", fraction, dim)
            y_noisy = shuffle_labels(y_train, fraction, rng)
            model = make_best_svm()
            model.fit(X_train, y_noisy)
            scores = evaluate(model, X_test, y_test)
            results[dataset_name][fraction].append(scores)
            print_scores(f"{dataset_name}, dim={dim}, {fraction:.0%} selected", scores)

# One figure per dataset and metric; each line is a shuffling level.
# Distinct markers/line styles keep overlapping lines visible.
markers = ["o", "s", "^", "D"]
linestyles = ["-", "--", "-.", ":"]
sizes = [11, 8, 6, 4]
for dataset_name in generators:
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, fraction in enumerate(NOISE_LEVELS):
            vals = [e[metric] for e in results[dataset_name][fraction]]
            ax.plot(DIMS, vals, marker=markers[k], ms=sizes[k], ls=linestyles[k],
                    label=f"{fraction:.0%} labels selected")
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("AUC" if metric == "auc" else metric.capitalize())
        ax.set_xticks(DIMS)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{dataset_name} dataset: "
                     f"{'AUC' if metric == 'auc' else metric} vs dimension")
        ax.legend(loc="lower left")
        fig.tight_layout()
        save_fig(fig, f"q6_{dataset_name.lower()}_{metric}_vs_dimension")
plt.show()
