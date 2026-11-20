"""
Shared helpers: evaluation metrics and plotting.
Use these in every question so the figures and numbers look consistent.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Colorblind-safe categorical palette (fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLASS_COLORS = PALETTE[:2]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def evaluate(model, X_test, y_test):
    """Return a dict with accuracy, F1 and (if possible) AUC."""
    y_pred = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    score_fn = getattr(model, "decision_function", None)
    if score_fn is not None:
        scores["auc"] = roc_auc_score(y_test, score_fn(X_test))
    return scores


def print_scores(name, scores):
    parts = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in scores.items())
    print(f"{name:30s} {parts}")


def plot_dataset(X, y, ax=None, title="", s=4, alpha=0.5):
    """Scatter plot of a 2D dataset colored by class."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    for c in (0, 1):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=s, alpha=alpha,
                   color=CLASS_COLORS[c], label=f"class {c}")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(markerscale=3, loc="upper right")
    return ax


def plot_decision_boundary(model, X, y, ax=None, title="", grid_steps=300):
    """Decision regions of a fitted 2D classifier + the data on top."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_steps),
                         np.linspace(y_min, y_max, grid_steps))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, levels=[-0.5, 0.5, 1.5],
                colors=CLASS_COLORS)
    plot_dataset(X, y, ax=ax, title=title)
    return ax


def plot_confusion(model, X_test, y_test, ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    return ax


def save_fig(fig, name):
    """Save into figures/<name>.png (use the question number as prefix)."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    return path
