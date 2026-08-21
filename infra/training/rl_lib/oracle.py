"""
VENDORED from data/oracle.py — see dataset.py in this package for why.
Keep this in sync with the real file; nothing here should diverge from it.
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

    def get_labels(self, indices) -> np.ndarray:
        """Returns the true labels for a batch of indices — every index must
        already be revealed (raises otherwise, same as reveal() on a repeat)."""
        for index in indices:
            if not self.is_revealed(index):
                raise ValueError(f"Index {index} has not been revealed yet.")
        return self._pool_y[indices]

    @property
    def num_revealed(self) -> int:
        return self._revealed_mask.sum()

    @property
    def revealed_mask(self) -> np.ndarray:
        """Boolean array, True where that pool index has been revealed — a vectorized
        alternative to calling is_revealed() in a per-index loop, e.g. in action_masks()."""
        return self._revealed_mask.copy()
