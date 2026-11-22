"""Q5 - Label noise / misclassification (20 pts). Owner: Mahima

TODO:
  - shuffle a percentage of the TRAINING labels only (original data,
    not augmented): 0%, 15%, 25%, 50%
  - retrain make_best_svm(), evaluate on the clean test set
  - plot performance vs % shuffled labels (both datasets), write conclusion
"""
import numpy as np
import matplotlib.pyplot as plt

from data import make_linear, make_circular, split, SEED
from best_model import make_best_svm
from utils import evaluate, print_scores, save_fig

NOISE_LEVELS = [0.0, 0.15, 0.25, 0.50]
rng = np.random.default_rng(SEED)

def shuffle_labels(y, fraction, rng):
    """Randomly permute labels for the selected fraction of training rows."""
    y_noisy = y.copy()
    n = int(round(fraction * len(y)))
    idx = rng.choice(len(y), size=n, replace=False)
    y_noisy[idx] = rng.permutation(y_noisy[idx])
    return y_noisy

datasets = {"Linear": make_linear(), "Circular": make_circular()}
results = {}
for dataset_name, (X, y) in datasets.items():
    X_train, X_test, y_train, y_test = split(X, y)
    results[dataset_name] = {}
    for fraction in NOISE_LEVELS:
        y_train_noisy = shuffle_labels(y_train, fraction, rng)
        model = make_best_svm()
        model.fit(X_train, y_train_noisy)
        scores = evaluate(model, X_test, y_test)
        results[dataset_name][fraction] = scores
        print_scores(f"{dataset_name}, {fraction:.0%} labels shuffled", scores)

# One plot per metric, with separate lines for the two datasets.
for metric in ["accuracy", "f1", "auc"]:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset_name in datasets:
        vals = [results[dataset_name][f][metric] for f in NOISE_LEVELS]
        ax.plot([100*f for f in NOISE_LEVELS], vals, marker="o", label=dataset_name)
    ax.set_xlabel("Training labels shuffled (%)")
    ax.set_ylabel(metric.upper() if metric == "auc" else metric.capitalize())
    ax.set_xticks([0, 15, 25, 50])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Effect of training-label noise on {metric.upper() if metric == 'auc' else metric}")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"q5_label_noise_{metric}")
plt.show()




