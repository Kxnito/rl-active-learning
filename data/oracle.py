"""
Simulates the "ask a human annotator" step. The Oracle is the only thing
that ever sees pool_y — the env asks it for a label by index, and it's the
Oracle's job to enforce that each index can only be revealed once and to
track how much of the labeling budget has been spent.
"""

import numpy as np


class Oracle:
    def __init__(self, pool_y: np.ndarray):
        self._pool_y = pool_y
        self._revealed_mask = np.zeros(len(pool_y), dtype=bool)

    def reveal(self, index: int) -> int:
        if index < 0 or index >= len(self._pool_y):
            raise IndexError(f"Index {index} out of range")
        if self._revealed_mask[index]:
            raise ValueError(f"Index {index} has already been revealed")
    
        self._revealed_mask[index] = True
        return self._pool_y[index]

    def is_revealed(self, index: int) -> bool:
        if index < 0 or index >= len(self._pool_y):
            raise IndexError(f"Index {index} out of range")
        return self._revealed_mask[index]

    @property
    def num_revealed(self) -> int:
        return self._revealed_mask.sum()
    
    @property
    def revealed_mask(self) -> np.ndarray:
        """Boolean array, True where that pool index has been revealed — a vectorized
        alternative to calling is_revealed() in a per-index loop, e.g. in action_masks()."""
        return self._revealed.copy()
