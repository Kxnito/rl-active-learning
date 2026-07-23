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
    """
    TODO(Person A): load via sklearn.datasets.load_breast_cancer(), shuffle,
    and split into seed/pool/val/test per the sizes above. Consider
    stratifying by class so the seed set isn't accidentally single-class.
    """
    raise NotImplementedError
