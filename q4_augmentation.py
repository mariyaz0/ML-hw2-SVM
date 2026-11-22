"""Q4 - Overfitting check via data augmentation (20 pts) + Bonus.

Design (state this in the report):
  - The best SVM (RBF) is trained ONCE on the clean original training set.
  - Each augmented dataset = 1000 points sampled (without replacement) from the
    clean held-out test set, plus independent Gaussian noise N(0, SD^2) on
    every feature. This is repeated K times (K >> 1000).
  - Metrics are computed per augmented dataset, then reported as mean +/- SD
    across the K repeats.
Bonus: sweep the noise SD and find where mean accuracy first drops below
THRESHOLD (reported both as the first tested SD and as an interpolated value).
"""
import numpy as np
import matplotlib.pyplot as plt

from data import make_circular, split, SEED
from best_model import make_best_svm
from utils import evaluate, plot_dataset, save_fig

LOW_SD = 0.1
HIGH_SD = 1.0
SAMPLE_SIZE = 1000
K = 10_000                      # k >> 1000
THRESHOLD = 0.70
BONUS_SDS = np.round(np.arange(0.0, 2.001, 0.05), 2)
BONUS_K = 200                   # repeats per SD in the bonus sweep
METRICS = ["accuracy", "f1", "auc"]

rng = np.random.default_rng(SEED)
X, y = make_circular()
X_train, X_test, y_train, y_test = split(X, y)

model = make_best_svm()
model.fit(X_train, y_train)     # clean original training data only


def noisy_batch(X_src, y_src, sd):
    idx = rng.choice(len(X_src), size=SAMPLE_SIZE, replace=False)
    Xb = X_src[idx] + rng.normal(0.0, sd, size=(SAMPLE_SIZE, X_src.shape[1]))
    return Xb, y_src[idx]


def repeated_scores(sd, k):
    """Return {metric: array of length k} - one score per augmented dataset."""
    out = {m: np.empty(k) for m in METRICS}
    for i in range(k):
        Xb, yb = noisy_batch(X_test, y_test, sd)
        s = evaluate(model, Xb, yb)
        for m in METRICS:
            out[m][i] = s[m]
    return out


# ---- Main experiment -------------------------------------------------------
clean = evaluate(model, X_test, y_test)
runs = {"Clean test": {m: np.array([clean[m]]) for m in METRICS}}
for label, sd in [("Low noise", LOW_SD), ("High noise", HIGH_SD)]:
    runs[label] = repeated_scores(sd, K)

print(f"\nK = {K} augmented datasets of {SAMPLE_SIZE} points each")
for label, sc in runs.items():
    print(f"{label:11s} " + "  ".join(
        f"{m}={sc[m].mean():.4f} +/- {sc[m].std():.4f}" for m in METRICS))

# ---- Figure: one representative batch per noise level ----------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (label, sd) in zip(axes, [("Low noise", LOW_SD), ("High noise", HIGH_SD)]):
    Xb, yb = noisy_batch(X_test, y_test, sd)
    plot_dataset(Xb, yb, ax=ax, s=5, alpha=0.35, title=f"{label} (SD={sd})")
fig.tight_layout()
save_fig(fig, "q4_augmented_samples")

# ---- Figure: mean metrics with SD error bars --------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(METRICS))
width = 0.24
for j, (label, sc) in enumerate(runs.items()):
    means = [sc[m].mean() for m in METRICS]
    sds = [sc[m].std() for m in METRICS]
    bars = ax.bar(x + (j - 1) * width, means, width, yerr=sds, capsize=3, label=label)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
ax.set_xticks(x, ["Accuracy", "F1", "AUC"])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score (mean ± SD over repeats)")
ax.set_title("SVM performance under Gaussian feature noise")
ax.legend(loc="lower left")
fig.tight_layout()
save_fig(fig, "q4_augmentation_metrics")

# ---- Bonus ------------------------------------------------------------------
bonus_mean, bonus_sd = [], []
for sd in BONUS_SDS:
    acc = repeated_scores(sd, BONUS_K)["accuracy"]
    bonus_mean.append(acc.mean())
    bonus_sd.append(acc.std())
bonus_mean = np.array(bonus_mean)
bonus_sd = np.array(bonus_sd)

below = np.where(bonus_mean < THRESHOLD)[0]
first_below = BONUS_SDS[below[0]] if len(below) else None
interp = None
if len(below) and below[0] > 0:
    i = below[0]
    x0, x1 = BONUS_SDS[i - 1], BONUS_SDS[i]
    y0, y1 = bonus_mean[i - 1], bonus_mean[i]
    interp = x0 + (y0 - THRESHOLD) * (x1 - x0) / (y0 - y1)

print(f"\nBonus: threshold = {THRESHOLD:.0%} accuracy")
for sd in (LOW_SD, HIGH_SD):
    i = int(np.argmin(np.abs(BONUS_SDS - sd)))
    print(f"  accuracy at SD={BONUS_SDS[i]:.2f}: {bonus_mean[i]:.4f}")
print("  first tested SD below threshold:", first_below)
print("  interpolated crossing SD:", None if interp is None else round(float(interp), 3))
if first_below is not None:
    i = below[0]
    print(f"  accuracy at SD={BONUS_SDS[i-1]:.2f}: {bonus_mean[i-1]:.4f}, "
          f"at SD={BONUS_SDS[i]:.2f}: {bonus_mean[i]:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(BONUS_SDS, bonus_mean, yerr=bonus_sd, marker="o", ms=4, capsize=2)
ax.axhline(THRESHOLD, linestyle="--", color="0.4", label=f"{THRESHOLD:.0%} threshold")
if interp is not None:
    ax.axvline(interp, linestyle=":", color="0.4", label=f"crossing: SD≈{interp:.2f}")
for sd, name in [(LOW_SD, "low"), (HIGH_SD, "high")]:
    ax.axvline(sd, linestyle="-", color="tab:green", alpha=0.25)
    ax.text(sd, 0.05, f" {name}", color="tab:green", fontsize=8)
ax.set_xlabel("Gaussian noise standard deviation")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Bonus: accuracy versus augmentation noise")
ax.legend()
fig.tight_layout()
save_fig(fig, "q4_bonus_accuracy_vs_noise")
plt.close(fig)