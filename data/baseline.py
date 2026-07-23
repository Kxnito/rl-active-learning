"""
The two baselines the RL agent gets compared against (project-context.md
Section 1). Both share the same Dataset/Oracle setup as the RL env so the
comparison is apples-to-apples — same seed set, same pool, same budget,
same val/test splits.

Each function should return a list/array of (labels_used, val_accuracy)
pairs — one point per label revealed — so eval/ can plot accuracy-per-label
curves against the RL agent's curve.
"""

from data.dataset import DatasetSplits
from data.oracle import Oracle


def run_random_sampling(splits: DatasetSplits, oracle: Oracle, budget: int):
    """
    TODO(Person A): at each step, pick a uniformly random not-yet-revealed
    pool index, reveal it, retrain the student model on seed + revealed
    labels so far, record (labels_used, val_accuracy). Repeat until budget
    is spent.
    """
    raise NotImplementedError


def run_uncertainty_sampling(splits: DatasetSplits, oracle: Oracle, budget: int):
    """
    TODO(Person A): at each step, pick the pool index the current student
    model is *least confident* about (e.g. smallest margin between top two
    predicted class probabilities), reveal it, retrain, record
    (labels_used, val_accuracy). This is the classic active learning
    heuristic the RL agent needs to beat.
    """
    raise NotImplementedError
