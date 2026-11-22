"""Q6 - High dimensionality (10 pts). Owner: Mahima

TODO:
  - generate datasets with dim = 5, 10, 15 (data.py supports `dim=`)
  - repeat the Q5 analysis (label noise) for each dimension
  - plot SVM performance as a function of dimension
"""
import numpy as np
import matplotlib.pyplot as plt

from data import make_linear, make_circular, split, SEED
from best_model import make_best_svm
from utils import evaluate, print_scores, save_fig

DIMS = [2, 5, 10, 15]
NOISE_LEVELS = [0.0, 0.15, 0.25, 0.50]
rng = np.random.default_rng(SEED)

def shuffle_labels(y, fraction, rng):
    y_noisy = y.copy()
    n = int(round(fraction * len(y)))
    idx = rng.choice(len(y), size=n, replace=False)
    y_noisy[idx] = rng.permutation(y_noisy[idx])
    return y_noisy

generators = {"Linear": make_linear, "Circular": make_circular}
results = {name: {noise: [] for noise in NOISE_LEVELS} for name in generators}

for dataset_name, generator in generators.items():
    for dim in DIMS:
        X, y = generator(dim=dim)
        X_train, X_test, y_train, y_test = split(X, y)
        for fraction in NOISE_LEVELS:
            y_train_noisy = shuffle_labels(y_train, fraction, rng)
            model = make_best_svm()
            model.fit(X_train, y_train_noisy)
            scores = evaluate(model, X_test, y_test)
            results[dataset_name][fraction].append(scores)
            print_scores(f"{dataset_name}, dim={dim}, {fraction:.0%} labels shuffled", scores)

# Plot accuracy, F1 and AUC against dimension. Each line is a label-noise level.
for dataset_name in generators:
    for metric in ["accuracy", "f1", "auc"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for fraction in NOISE_LEVELS:
            vals = [entry[metric] for entry in results[dataset_name][fraction]]
            ax.plot(DIMS, vals, marker="o", label=f"{fraction:.0%} labels shuffled")
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel(metric.upper() if metric == "auc" else metric.capitalize())
        ax.set_xticks(DIMS)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{dataset_name} dataset: {metric.upper() if metric == 'auc' else metric} vs dimension")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, f"q6_{dataset_name.lower()}_{metric}_vs_dimension")
plt.show()
