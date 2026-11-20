"""
Dataset simulation for HW2 (Question 1, also reused in Q2-Q6).

Both generators work for any number of input dimensions (dim=2 by default),
so Question 6 (high dimensionality) can reuse them directly.

"""
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 67          # shared seed for the whole project - do not change
N_SAMPLES = 10_000
TEST_SIZE = 0.3


def make_linear(n=N_SAMPLES, dim=2, separation=12.0, std=1.5, seed=SEED):
    """Two Gaussian clusters far from each other (linearly separable).

    Cluster centers are at -separation/2 and +separation/2 along the
    direction (1, 1, ..., 1) / sqrt(dim), so the distance between the
    centers is `separation` for any `dim`.
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    direction = np.ones(dim) / np.sqrt(dim)
    c0 = -direction * separation / 2
    c1 = direction * separation / 2
    X0 = rng.normal(c0, std, size=(n0, dim))
    X1 = rng.normal(c1, std, size=(n1, dim))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    return _shuffle(X, y, rng)


def make_circular(n=N_SAMPLES, dim=2, inner_std=0.5, r_outer=3.0,
                  noise=0.3, seed=SEED):
    """One cluster inside a ring/shell of the other cluster.

    Class 0: Gaussian cluster at the origin with std `inner_std`.
    Class 1: points on a ring (2D) / spherical shell (>2D) of radius
             `r_outer`; `noise` is the std of the radius (ring thickness).
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    # inner cluster
    X0 = rng.normal(0, inner_std, size=(n0, dim))

    # outer ring / shell: random direction on the unit sphere * noisy radius
    directions = rng.normal(size=(n1, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = r_outer + rng.normal(0, noise, size=(n1, 1))
    X1 = directions * radii

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    return _shuffle(X, y, rng)


def split(X, y, test_size=TEST_SIZE, seed=SEED):
    """Shared train/test split (stratified) - use this everywhere."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)


def _shuffle(X, y, rng):
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


if __name__ == "__main__":
    for name, fn in [("linear", make_linear), ("circular", make_circular)]:
        for d in (2, 5, 10, 15):
            X, y = fn(dim=d)
            print(f"{name:8s} dim={d:2d}  X={X.shape}  class balance={y.mean():.2f}")
