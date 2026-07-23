"""
Simulates the "ask a human annotator" step. The Oracle is the only thing
that ever sees pool_y — the env asks it for a label by index, and it's the
Oracle's job to enforce that each index can only be revealed once and to
track how much of the labeling budget has been spent.
"""

import numpy as np


class Oracle:
    def __init__(self, pool_y: np.ndarray):
        """
        TODO(Person A): store pool_y privately and set up whatever
        bookkeeping reveal()/is_revealed() need (e.g. a revealed-mask or
        a set of queried indices).
        """
        raise NotImplementedError

    def reveal(self, index: int) -> int:
        """
        TODO(Person A): return the true label for pool[index] and mark it
        as revealed. Raise if index was already revealed or is out of
        range — the env should never be calling this on an invalid index,
        so a loud failure here is more useful than a silent one.
        """
        raise NotImplementedError

    def is_revealed(self, index: int) -> bool:
        """TODO(Person A): used by env.action_masks() to mark invalid actions."""
        raise NotImplementedError

    @property
    def num_revealed(self) -> int:
        """TODO(Person A): used by env state (labels used so far) and to check the budget."""
        raise NotImplementedError
