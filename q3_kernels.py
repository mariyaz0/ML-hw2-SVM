"""Q3 - SVM kernels on the circular dataset. Owner: Mariya

Main comparison: RBF kernel vs polynomial kernel (sklearn default: degree 3,
coef0 = 0), plus the same polynomial kernel with coef0 = 1.
Extra analysis: accuracy of the polynomial kernel as a function of its degree
for coef0 = 0 and coef0 = 1, which explains WHY the default polynomial kernel
fails here and how coef0 fixes it.

Figures (saved to figures/):
  q3_decision_boundaries.png - decision boundaries of all kernels (test set)
  q3_confusion_matrices.png  - confusion matrices of all kernels
  q3_roc.png                 - ROC curves of all kernels
  q3_poly_degree.png         - polynomial kernel accuracy vs degree
"""
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve

from data import make_circular, split
from utils import (evaluate, print_scores, plot_decision_boundary,
                   plot_confusion, save_fig, PALETTE)

X, y = make_circular()
X_train, X_test, y_train, y_test = split(X, y)

kernels = {
    "RBF": SVC(kernel="rbf", gamma="scale", C=1.0),
    "Polynomial (degree 3)": SVC(kernel="poly", degree=3, gamma="scale", C=1.0),
    "Polynomial (degree 3, coef0=1)": SVC(kernel="poly", degree=3, coef0=1,
                                          gamma="scale", C=1.0),
}

# ---------------------------------------------------------------------------
# 1. Fit, evaluate, time
# ---------------------------------------------------------------------------
results = {}
for name, model in kernels.items():
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    scores = evaluate(model, X_test, y_test)
    scores["n_support_vectors"] = int(model.n_support_.sum())
    scores["fit_time_s"] = fit_time
    results[name] = scores
    print_scores(name, scores)

# ---------------------------------------------------------------------------
# 2. Decision boundaries
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(kernels), figsize=(5.5 * len(kernels), 5.5))
for ax, (name, model) in zip(axes, kernels.items()):
    plot_decision_boundary(model, X_test, y_test, ax=ax,
                           title=f"{name}\n(acc = {results[name]['accuracy']:.2f})")
fig.tight_layout()
save_fig(fig, "q3_decision_boundaries")

# ---------------------------------------------------------------------------
# 3. Confusion matrices
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(kernels), figsize=(4.2 * len(kernels), 4))
for ax, (name, model) in zip(axes, kernels.items()):
    plot_confusion(model, X_test, y_test, ax=ax, title=name.replace(", ", ",\n"))
    ax.grid(False)  # no grid lines across the matrix cells
fig.tight_layout()
save_fig(fig, "q3_confusion_matrices")

# ---------------------------------------------------------------------------
# 4. ROC curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5.5))
line_styles = ["-", "-", "--"]
for i, (name, model) in enumerate(kernels.items()):
    fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
    ax.plot(fpr, tpr, color=PALETTE[i], lw=2, ls=line_styles[i],
            label=f"{name} (AUC = {results[name]['auc']:.2f})")
ax.plot([0, 1], [0, 1], color="0.5", ls=":", lw=1, label="Chance level")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves, circular dataset")
ax.legend(loc="lower right", fontsize=8)
save_fig(fig, "q3_roc")

# ---------------------------------------------------------------------------
# 5. Polynomial degree sweep.
#    The circle boundary is x1^2 + x2^2 = r^2, i.e. a degree-2 function.
#    With coef0 = 0 (sklearn default) the polynomial kernel (g * <x, x'>)^d
#    contains ONLY monomials of degree exactly d, so odd degrees cannot
#    express x1^2 + x2^2 and fail, while even degrees succeed.
#    With coef0 = 1 the kernel (g * <x, x'> + 1)^d contains ALL monomials
#    of degree <= d, so every degree >= 2 includes x1^2 + x2^2.
# ---------------------------------------------------------------------------
degrees = list(range(1, 7))
poly_acc = {0: [], 1: []}          # coef0 -> accuracy for each degree
for c0 in (0, 1):
    for d in degrees:
        m = SVC(kernel="poly", degree=d, coef0=c0, gamma="scale")
        m.fit(X_train, y_train)
        poly_acc[c0].append(m.score(X_test, y_test))
        print(f"poly degree {d}, coef0={c0}: accuracy = {poly_acc[c0][-1]:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(degrees, poly_acc[0], marker="o", ms=8, lw=2, color=PALETTE[1],
        label="Polynomial, coef0 = 0 (default)")
ax.plot(degrees, poly_acc[1], marker="s", ms=8, lw=2, ls="--",
        color=PALETTE[2], label="Polynomial, coef0 = 1")
ax.axhline(results["RBF"]["accuracy"], color=PALETTE[0], lw=2, ls=":",
           label="RBF kernel")
ax.axhline(0.5, color="0.4", lw=1, ls=":", label="Chance level")
ax.set_xlabel("Polynomial degree")
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.4, 1.05)
ax.set_title("Polynomial kernel: accuracy vs degree")
ax.legend(fontsize=8, loc="lower right")
save_fig(fig, "q3_poly_degree")

# ---------------------------------------------------------------------------
# 6. Table for the report
# ---------------------------------------------------------------------------
print("\nTable 2 (copy to the report):")
print(f"{'Kernel':32s} {'Accuracy':>9s} {'F1':>6s} {'AUC':>6s} {'#SV':>6s} "
      f"{'fit time, s':>11s}")
for name, s in results.items():
    print(f"{name:32s} {s['accuracy']:9.3f} {s['f1']:6.3f} {s['auc']:6.3f} "
          f"{s['n_support_vectors']:6d} {s['fit_time_s']:11.2f}")

best_f1 = max(s["f1"] for s in results.values())
best = [name for name, s in results.items() if s["f1"] == best_f1]
print(f"\nBest F1 = {best_f1:.3f}: {', '.join(best)}")

plt.show()