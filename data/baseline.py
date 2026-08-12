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
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple


class StudentModel:
    def __init__(self, splits: DatasetSplits, revealed_seed_indices: List[int], revealed_pool_indices: List[int], oracle: Oracle):
        self.splits = splits
        self.oracle = oracle
        self.revealed_seed_indices = revealed_seed_indices
        self.revealed_pool_indices = revealed_pool_indices
        self.model = LogisticRegression(max_iter=1000)

    def train(self):
        # Combine seed and revealed pool data
        X_seed = self.splits.seed_X[self.revealed_seed_indices]
        y_seed = self.splits.seed_y[self.revealed_seed_indices]

        X_pool = self.splits.pool_X[self.revealed_pool_indices]
        y_pool = self.oracle.get_labels(self.revealed_pool_indices)

        X_train = np.vstack([X_seed, X_pool])
        y_train = np.concatenate([y_seed, y_pool])

        if len(np.unique(y_train)) < 2:
            return

        self.model.fit(X_train, y_train)

    def predict_proba(self, pool_indices: List[int]) -> np.ndarray:
        X_pool = self.splits.pool_X[pool_indices]
        return self.model.predict_proba(X_pool)

    def evaluate_val_accuracy(self) -> float:
        X_val = self.splits.val_X
        y_val = self.splits.val_y
        try:
            y_pred = self.model.predict(X_val)
            return np.mean(y_pred == y_val)
        except Exception:
            return 0.0

def run_random_sampling(splits: DatasetSplits, oracle: Oracle, budget: int) -> List[Tuple[int, float]]:
    revealed_seed_indices = list(range(len(splits.seed_X)))
    revealed_pool_indices = []
    results = []

    pool_indices = list(range(len(splits.pool_X)))

    for _ in range(budget):
        unrevealed = [idx for idx in pool_indices if idx not in revealed_pool_indices]
        if not unrevealed:
            break

        chosen = np.random.choice(unrevealed)
        oracle.reveal(chosen)
        revealed_pool_indices.append(chosen)

        model = StudentModel(splits, revealed_seed_indices, revealed_pool_indices, oracle)
        model.train()

        acc = model.evaluate_val_accuracy()
        results.append((len(revealed_seed_indices) + len(revealed_pool_indices), acc))

    return results

def run_uncertainty_sampling(splits: DatasetSplits, oracle: Oracle, budget: int) -> List[Tuple[int, float]]:
    revealed_seed_indices = list(range(len(splits.seed_X)))
    revealed_pool_indices = []
    results = []

    pool_indices = list(range(len(splits.pool_X)))

    for _ in range(budget):
        model = StudentModel(splits, revealed_seed_indices, revealed_pool_indices, oracle)
        model.train()

        unrevealed = [idx for idx in pool_indices if idx not in revealed_pool_indices]
        if not unrevealed:
            break

        probs = model.predict_proba(unrevealed)

        margins = []
        for p in probs:
            sorted_p = sorted(p, reverse=True)
            margin = sorted_p[0] - sorted_p[1]
            margins.append(margin)

        min_margin_idx = np.argmin(margins)
        chosen = unrevealed[min_margin_idx]

        oracle.reveal(chosen)
        revealed_pool_indices.append(chosen)

        acc = model.evaluate_val_accuracy()
        results.append((len(revealed_seed_indices) + len(revealed_pool_indices), acc))

    return results