"""
Loads Breast Cancer Wisconsin and splits it into the pools the active
learning loop needs. See project-context.md Section 1 for the RL framing
this feeds into.

Splits to produce:
  - seed set:       a small number of already-labeled examples the student
                     model starts from (episode 0 state)
  - unlabeled pool:  everything the RL agent can choose to query via Oracle
  - val set:         held out, used to compute the reward (accuracy delta)
                     after each label is revealed — must NOT leak into
                     training
  - test set:        held out, used only for final reported numbers —
                     must NOT be touched during training or reward
                     computation

sklearn.datasets.load_breast_cancer() ships the data + labels together;
splitting off the "labels" into data/oracle.py's Oracle is what turns this
static dataset into an active learning problem.
"""

from dataclasses import dataclass
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np


@dataclass
class DatasetSplits:
    seed_X: np.ndarray
    seed_y: np.ndarray
    pool_X: np.ndarray
    pool_y: np.ndarray  # hand this to Oracle only — the env/agent should never read it directly
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray


def load_dataset(seed_size: int, val_size: int, test_size: int, random_state: int = 0) -> DatasetSplits:
    """Splits Breast Cancer Wisconsin into seed/pool/val/test per DatasetSplits."""
    
    data = load_breast_cancer()
    X, y = data.data, data.target

    # cut 1: carve off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # cut 2: carve off the val set from what's left
    X_temp2, X_val, y_temp2, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )

    # cut 3: split what's left into seed (small) and pool (the rest)
    X_seed, X_pool, y_seed, y_pool = train_test_split(
        X_temp2, y_temp2, train_size=seed_size, stratify=y_temp2, random_state=random_state
    )

    return DatasetSplits(
        seed_X=X_seed, seed_y=y_seed,
        pool_X=X_pool, pool_y=y_pool,
        val_X=X_val, val_y=y_val,
        test_X=X_test, test_y=y_test,
    )

