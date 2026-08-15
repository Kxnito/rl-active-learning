import numpy as np
import pytest
import unittest

from data.dataset import load_dataset
import data.oracle as Oracle

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
