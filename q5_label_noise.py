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


def shuffle_labels(y, fraction, rng):
    """Randomly permute the labels of `fraction` of the points."""
    y_noisy = y.copy()
    n = int(round(fraction * len(y)))
    idx = rng.choice(len(y), size=n, replace=False)
    y_noisy[idx] = rng.permutation(y_noisy[idx])
    return y_noisy


rng = np.random.default_rng(SEED)

# TODO: fill in
