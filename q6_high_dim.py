"""Q6 - High dimensionality (10 pts). Owner: Mahima

TODO:
  - generate datasets with dim = 5, 10, 15 (data.py supports `dim=`)
  - repeat the Q5 analysis (label noise) for each dimension
  - plot SVM performance as a function of dimension
"""
import numpy as np
import matplotlib.pyplot as plt

from data import make_linear, make_circular, split, SEED
from best_model import make_best_svm
from utils import evaluate, print_scores, save_fig

DIMS = [2, 5, 10, 15]

# TODO: fill in
