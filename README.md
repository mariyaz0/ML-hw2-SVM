# HW2 — SVM (Machine Learning & Neural Networks for Neuroscience, BIU)

Authors: Mariya Zastenskaya, Mahima Chaudhary

## Setup
```bash
pip install -r requirements.txt
```

## How to run
Each question is a standalone script; figures are saved to `figures/`.
```bash
python q1_datasets.py
python q2_perceptron_vs_svm.py
python q3_kernels.py
python q4_augmentation.py
python q5_label_noise.py
python q6_high_dim.py
```

## Structure
| File | Content |
|---|---|
| `data.py` | dataset generators (any dimension), shared seed and train/test split |
| `utils.py` | metrics (accuracy, F1, AUC), scatter / decision boundary / confusion matrix plots |
| `best_model.py` | the best SVM chosen in Q3 (used by Q4–Q6) |
| `q1_datasets.py` | Q1 — dataset simulation |
| `q2_perceptron_vs_svm.py` | Q2 — perceptron vs linear SVM |
| `q3_kernels.py` | Q3 — SVM kernels |
| `q4_augmentation.py` | Q4 — data augmentation + bonus |
| `q5_label_noise.py` | Q5 — label noise |
| `q6_high_dim.py` | Q6 — high dimensionality |

## Conventions
- Random seed: `SEED = 67` (in `data.py`) — everywhere.
- Always split with `data.split()` so train/test sets are identical across questions.
- Save figures with `utils.save_fig(fig, "qN_description")`.
- Edit only your own files to avoid merge conflicts; `git pull` before every `git push`.
