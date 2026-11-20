"""
The "best-fitted SVM" chosen in Q3. Q4, Q5 and Q6 import it from here,
so if Q3 picks a different kernel, change it ONLY in this file.
"""
from sklearn.svm import SVC

BEST_KERNEL = "rbf"   # <- update after Q3 if needed


def make_best_svm():
    return SVC(kernel=BEST_KERNEL, gamma="scale", C=1.0)
