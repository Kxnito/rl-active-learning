"""
VENDORED from data/dataset.py — SageMaker's SKLearn estimator only ships
infra/training/'s contents to the container, so this package carries a
minimal, self-contained copy of what the real active-learning loop needs.

If data/dataset.py's DatasetSplits fields change, update this to match.

Only the DatasetSplits container is needed here, not load_dataset() itself
— infra/upload_data.py already runs the real load_dataset() locally and
uploads the resulting arrays as a .npz, so this side just needs somewhere
to hold them after infra/training/train.py loads that .npz back.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetSplits:
    seed_X: np.ndarray
    seed_y: np.ndarray
    pool_X: np.ndarray
    pool_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
