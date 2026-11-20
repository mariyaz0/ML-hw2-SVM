"""Q2 - Perceptron vs linear SVM on both datasets. Owner: Mariya

Figures (saved to figures/):
  q2_decision_boundaries.png  - 2x2 grid: model x dataset, on the test set
  q2_perceptron_vs_svm_lines.png - linear dataset: 10 perceptrons (different
                                 random seeds) vs the single max-margin SVM line
  q2_metrics.png              - accuracy / F1 / AUC for every model x dataset
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

from data import make_linear, make_circular, split, SEED
from utils import (evaluate, print_scores, plot_decision_boundary,
                   plot_dataset, save_fig, PALETTE)

datasets = {"Linear": make_linear(), "Circular": make_circular()}
models = {
    "Perceptron": lambda: Perceptron(random_state=SEED),
    "Linear SVM": lambda: SVC(kernel="linear"),
}

# ---------------------------------------------------------------------------
# 1. Train both models on both datasets, evaluate on the test set
# ---------------------------------------------------------------------------
results = {}        # (model, dataset) -> scores
fitted = {}         # (model, dataset) -> fitted model
splits = {}         # dataset -> (X_train, X_test, y_train, y_test)

for d_name, (X, y) in datasets.items():
    X_train, X_test, y_train, y_test = split(X, y)
    splits[d_name] = (X_train, X_test, y_train, y_test)
    for m_name, make_model in models.items():
        model = make_model().fit(X_train, y_train)
        fitted[(m_name, d_name)] = model
        results[(m_name, d_name)] = evaluate(model, X_test, y_test)
        print_scores(f"{m_name} / {d_name}", results[(m_name, d_name)])

# ---------------------------------------------------------------------------
# 2. Decision boundaries (2x2)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, d_name in enumerate(datasets):
    _, X_test, _, y_test = splits[d_name]
    for j, m_name in enumerate(models):
        acc = results[(m_name, d_name)]["accuracy"]
        plot_decision_boundary(fitted[(m_name, d_name)], X_test, y_test,
                               ax=axes[i, j],
                               title=f"{m_name}, {d_name.lower()} data "
                                     f"(acc = {acc:.2f})")
fig.tight_layout()
save_fig(fig, "q2_decision_boundaries")

# ---------------------------------------------------------------------------
# 3. Why SVM differs from the perceptron: on separable data the perceptron
#    stops at ANY separating line (depends on the seed / order of samples),
#    the SVM finds the single line with the maximal margin.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = splits["Linear"]
fig, ax = plt.subplots(figsize=(6.5, 6.5))
plot_dataset(X_train, y_train, ax=ax, alpha=0.25,
             title="Linear dataset: 10 perceptrons vs linear SVM")
xs = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 200)


def line_y(w, b, x):
    return -(w[0] * x + b) / w[1]


for k in range(10):
    p = Perceptron(random_state=k, shuffle=True).fit(X_train, y_train)
    ax.plot(xs, line_y(p.coef_[0], p.intercept_[0], xs), color="0.35",
            lw=1, alpha=0.7, label="Perceptron (10 seeds)" if k == 0 else None)

svm = fitted[("Linear SVM", "Linear")]
w, b = svm.coef_[0], svm.intercept_[0]
ax.plot(xs, line_y(w, b, xs), color=PALETTE[2], lw=2.5,
        label="SVM decision boundary")
for off in (-1, 1):  # margins: w.x + b = +-1
    ax.plot(xs, line_y(w, b - off, xs), color=PALETTE[2], lw=1.5, ls="--",
            label="SVM margin" if off == 1 else None)
sv = svm.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="black",
           lw=1.5, label=f"Support vectors ({len(sv)})", zorder=5)
ax.set_ylim(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5)
ax.legend(loc="upper left", fontsize=8, markerscale=1)
save_fig(fig, "q2_perceptron_vs_svm_lines")

# ---------------------------------------------------------------------------
# 4. Metrics bar chart
# ---------------------------------------------------------------------------
metrics = ["accuracy", "f1", "auc"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
width = 0.38
for ax, d_name in zip(axes, datasets):
    x = np.arange(len(metrics))
    for j, m_name in enumerate(models):
        vals = [results[(m_name, d_name)][m] for m in metrics]
        bars = ax.bar(x + (j - 0.5) * width, vals, width * 0.95,
                      color=PALETTE[j], label=m_name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(0.5, color="0.4", lw=1, ls=":", label="Chance level")
    ax.set_xticks(x, [m.upper() if m == "auc" else m.capitalize()
                      for m in metrics])
    ax.set_title(f"{d_name} dataset")
    ax.set_ylim(0, 1.1)
axes[0].set_ylabel("Test score")
axes[0].legend(loc="lower left", fontsize=8)
fig.tight_layout()
save_fig(fig, "q2_metrics")

# ---------------------------------------------------------------------------
# 5. Table for the report (copy into Table 1)
# ---------------------------------------------------------------------------
print("\nTable 1 (copy to the report):")
print(f"{'Model':12s} {'Dataset':9s} {'Accuracy':>9s} {'F1':>6s} {'AUC':>6s}")
for (m_name, d_name), s in results.items():
    print(f"{m_name:12s} {d_name:9s} {s['accuracy']:9.3f} {s['f1']:6.3f} "
          f"{s['auc']:6.3f}")

plt.show()

