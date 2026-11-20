"""Q1 - Dataset simulation (15 pts). Owner: Mariya"""
import matplotlib.pyplot as plt
from data import make_linear, make_circular
from utils import plot_dataset, save_fig

X_lin, y_lin = make_linear()
X_circ, y_circ = make_circular()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_dataset(X_lin, y_lin, ax=axes[0], title="(a) Linear dataset")
plot_dataset(X_circ, y_circ, ax=axes[1], title="(b) Circular dataset")
fig.tight_layout()
save_fig(fig, "q1_datasets")
plt.show()
