"""Q3 - SVM kernels on the circular dataset (15 pts). Owner: Mariya

TODO:
  - compare two kernels, e.g. SVC(kernel="rbf") vs SVC(kernel="poly", degree=2)
  - decision boundaries + confusion matrices + accuracy / F1 / AUC
  - decide which kernel is best -> update BEST_KERNEL in best_model.py
    and tell your partner (Q4, Q5 use it)
"""
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from data import make_circular, split
from utils import (evaluate, print_scores, plot_decision_boundary,
                   plot_confusion, save_fig)

X, y = make_circular()
X_train, X_test, y_train, y_test = split(X, y)

# TODO: fill in
