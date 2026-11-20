"""Q2 - Perceptron vs linear SVM on both datasets (20 pts). Owner: Mariya

TODO:
  - for each dataset (linear, circular): split -> fit Perceptron and
    SVC(kernel="linear") (or LinearSVC) -> evaluate
  - plot decision boundaries (2x2 grid: model x dataset)
  - bar plot / table of accuracy, F1, AUC
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

from data import make_linear, make_circular, split, SEED
from utils import evaluate, print_scores, plot_decision_boundary, save_fig

datasets = {"linear": make_linear(), "circular": make_circular()}
models = {
    "Perceptron": lambda: Perceptron(random_state=SEED),
    "Linear SVM": lambda: SVC(kernel="linear"),
}

# TODO: fill in
