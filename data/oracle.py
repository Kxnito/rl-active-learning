"""
Simulates the "ask a human annotator" step. The Oracle is the only thing
that ever sees pool_y — the env asks it for a label by index, and it's the
Oracle's job to enforce that each index can only be revealed once and to
track how much of the labeling budget has been spent.
"""

import numpy as np


class Oracle:
    def __init__(self, pool_y: np.ndarray):
        """Stores pool_y privately and initializes the revealed-index mask."""
        self._pool_y = pool_y
        self._revealed = np.zeros(len(pool_y), dtype=bool)  # Initialize a boolean array to track revealed indices

    def _check_in_range(self, index: int) -> None:
        if index < 0 or index >= len(self._pool_y):
            raise IndexError(f"Index {index} is out of range.")

    def reveal(self, index: int) -> int:
        """Returns pool_y[index] and marks it revealed; raises if already revealed or out of range."""
        self._check_in_range(index)
        if self._revealed[index]:
            raise ValueError(f"Index {index} has already been revealed.")
        self._revealed[index] = True
        return self._pool_y[index]

    def is_revealed(self, index: int) -> bool:
        """Whether index has already been revealed — used by env.action_masks()."""
        self._check_in_range(index)
        return bool(self._revealed[index])

    @property
    def num_revealed(self) -> int:
        """Number of indices revealed so far — used by env state and to check the budget."""
        return int(self._revealed.sum())

    @property
    def revealed_mask(self) -> np.ndarray:
        """Boolean array, True where that pool index has been revealed — a vectorized
        alternative to calling is_revealed() in a per-index loop, e.g. in action_masks()."""
        return self._revealed.copy()
