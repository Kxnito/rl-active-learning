import unittest
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


class TestOracle(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 0, 1, 1])
        self.oracle = Oracle(self.labels)

    def test_initial_state(self):
        # Initially, no labels revealed
        self.assertEqual(self.oracle.num_revealed, 0)
        for i in range(len(self.labels)):
            self.assertFalse(self.oracle.is_revealed(i))

    def test_reveal_valid_index(self):
        label = self.oracle.reveal(2)
        self.assertEqual(label, self.labels[2])
        self.assertTrue(self.oracle.is_revealed(2))
        self.assertEqual(self.oracle.num_revealed, 1)

    def test_reveal_twice_raises(self):
        self.oracle.reveal(1)
        with self.assertRaises(ValueError):
            self.oracle.reveal(1)

    def test_reveal_out_of_bounds_raises(self):
        with self.assertRaises(IndexError):
            self.oracle.reveal(-1)
        with self.assertRaises(IndexError):
            self.oracle.reveal(len(self.labels))

    def test_is_revealed_out_of_bounds_raises(self):
        with self.assertRaises(IndexError):
            self.oracle.is_revealed(-1)
        with self.assertRaises(IndexError):
            self.oracle.is_revealed(len(self.labels))

    def test_num_revealed_counts_correctly(self):
        self.assertEqual(self.oracle.num_revealed, 0)
        self.oracle.reveal(0)
        self.oracle.reveal(4)
        self.assertEqual(self.oracle.num_revealed, 2)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
