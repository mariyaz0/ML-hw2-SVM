"""Q4 - Overfitting check via data augmentation (20 pts). Owner: Mahima

TODO:
  a. sample 1000 points from the original (training) data
  b. add Gaussian noise - two options: LOW_SD and HIGH_SD
  c. repeat k times (k >> 1000)
  - train make_best_svm() on the original data, evaluate on augmented sets
  - scatter plots of low- vs high-noise augmented data
  - metrics (accuracy / F1 / AUC) for both noise levels, explain generalization
Bonus: pick a threshold (e.g. 70% accuracy) and find where performance
       drops below it for each augmentation.
"""
import numpy as np
import matplotlib.pyplot as plt

from data import make_circular, split, SEED
from best_model import make_best_svm
from utils import evaluate, print_scores, plot_dataset, save_fig

LOW_SD = 0.1
HIGH_SD = 1.0
SAMPLE_SIZE = 1000
K = 2000
THRESHOLD = 0.70
BONUS_SDS = np.arange(0.0, 2.01, 0.1)

rng = np.random.default_rng(SEED)
X, y = make_circular()
X_train, X_test, y_train, y_test = split(X, y)

# Fit only on clean original training data
model = make_best_svm()
model.fit(X_train, y_train)

def make_augmented_evaluation_set(X_source, y_source, noise_sd, k=K):
    """Create k noisy batches, each sampled from the original training set."""
    xs, ys = [], []
    for _ in range(k):
        idx = rng.choice(len(X_source), size=SAMPLE_SIZE, replace=True)
        batch = X_source[idx] + rng.normal(0.0, noise_sd, X_source[idx].shape)
        xs.append(batch)
        ys.append(y_source[idx])
    return np.concatenate(xs), np.concatenate(ys)

results = {}
for label, sd in [("Low noise", LOW_SD), ("High noise", HIGH_SD)]:
    X_aug, y_aug = make_augmented_evaluation_set(X_train, y_train, sd)
    scores = evaluate(model, X_aug, y_aug)
    results[label] = scores
    print_scores(label, scores)

# Visualize one representative batch at each noise level.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (label, sd) in zip(axes, [("Low noise", LOW_SD), ("High noise", HIGH_SD)]):
    idx = rng.choice(len(X_train), size=SAMPLE_SIZE, replace=True)
    X_batch = X_train[idx] + rng.normal(0.0, sd, X_train[idx].shape)
    plot_dataset(X_batch, y_train[idx], ax=ax, s=5, alpha=0.35,
                 title=f"{label} (SD={sd})")
fig.tight_layout()
save_fig(fig, "q4_augmented_samples")

# Compare performance on clean held-out data and noisy augmented data.
metric_names = ["accuracy", "f1", "auc"]
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ["Clean test", "Low noise", "High noise"]
score_sets = [evaluate(model, X_test, y_test), results["Low noise"], results["High noise"]]
x = np.arange(len(metric_names))
width = 0.24
for j, (name, scores) in enumerate(zip(labels, score_sets)):
    vals = [scores[m] for m in metric_names]
    bars = ax.bar(x + (j - 1) * width, vals, width, label=name)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks(x, ["Accuracy", "F1", "AUC"])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("SVM performance under Gaussian feature noise")
ax.legend()
fig.tight_layout()
save_fig(fig, "q4_augmentation_metrics")

# Bonus
bonus_accuracy = []
for sd in BONUS_SDS:
    X_aug, y_aug = make_augmented_evaluation_set(X_train, y_train, sd, k=50)
    bonus_accuracy.append(evaluate(model, X_aug, y_aug)["accuracy"])
below = [sd for sd, acc in zip(BONUS_SDS, bonus_accuracy) if acc < THRESHOLD]
crossing = below[0] if below else None
print(f"\nBonus threshold: {THRESHOLD:.0%} accuracy")
print("First tested SD below threshold:", crossing if crossing is not None else "not reached")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(BONUS_SDS, bonus_accuracy, marker="o")
ax.axhline(THRESHOLD, linestyle="--", color="0.4", label=f"{THRESHOLD:.0%} threshold")
if crossing is not None:
    ax.axvline(crossing, linestyle=":", color="0.4", label=f"first below: SD={crossing:.1f}")
ax.set_xlabel("Gaussian noise standard deviation")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Bonus: accuracy versus augmentation noise")