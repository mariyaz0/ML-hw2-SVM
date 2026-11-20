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

rng = np.random.default_rng(SEED)
X, y = make_circular()
X_train, X_test, y_train, y_test = split(X, y)

# TODO: fill in
