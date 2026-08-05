import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from data.dataset import load_dataset

_raw = load_breast_cancer()
TOTAL_SAMPLES = len(_raw.target)
OVERALL_POSITIVE_RATE = _raw.target.mean()  # class balance in the full dataset


@pytest.fixture
def splits():
    return load_dataset(seed_size=20, val_size=50, test_size=100, random_state=0)


def test_split_sizes_match_requested_sizes(splits):
    assert splits.seed_X.shape[0] == 20
    assert splits.val_X.shape[0] == 50
    assert splits.test_X.shape[0] == 100
    assert splits.pool_X.shape[0] == TOTAL_SAMPLES - 20 - 50 - 100


def test_splits_sum_to_total_dataset_size(splits):
    total = (
        splits.seed_X.shape[0]
        + splits.pool_X.shape[0]
        + splits.val_X.shape[0]
        + splits.test_X.shape[0]
    )
    assert total == TOTAL_SAMPLES


def test_X_and_y_lengths_match_within_each_split(splits):
    assert splits.seed_X.shape[0] == splits.seed_y.shape[0]
    assert splits.pool_X.shape[0] == splits.pool_y.shape[0]
    assert splits.val_X.shape[0] == splits.val_y.shape[0]
    assert splits.test_X.shape[0] == splits.test_y.shape[0]


def test_stratification_preserves_class_proportions():
    splits = load_dataset(seed_size=20, val_size=100, test_size=100, random_state=0)

    for split_y in [splits.val_y, splits.test_y]:
        split_rate = split_y.mean()
        assert abs(split_rate - OVERALL_POSITIVE_RATE) < 0.05


def test_same_random_state_is_reproducible():
    splits_a = load_dataset(seed_size=20, val_size=50, test_size=100, random_state=42)
    splits_b = load_dataset(seed_size=20, val_size=50, test_size=100, random_state=42)
    np.testing.assert_array_equal(splits_a.seed_y, splits_b.seed_y)
    np.testing.assert_array_equal(splits_a.test_y, splits_b.test_y)


def test_splits_are_disjoint(splits):
    """No sample should end up in more than one split — guards against a
    logic error silently double-counting or dropping rows."""
    row_sets = [
        {tuple(row) for row in X}
        for X in [splits.seed_X, splits.pool_X, splits.val_X, splits.test_X]
    ]
    for i in range(len(row_sets)):
        for j in range(i + 1, len(row_sets)):
            assert row_sets[i].isdisjoint(row_sets[j])


def test_sizes_exceeding_dataset_raises():
    """seed + val + test (300+300+300) exceeds the 569 total — should fail
    loudly rather than silently returning a bad/empty split."""
    with pytest.raises(ValueError):
        load_dataset(seed_size=300, val_size=300, test_size=300, random_state=0)


def test_zero_val_size_raises():
    with pytest.raises(ValueError):
        load_dataset(seed_size=20, val_size=0, test_size=100, random_state=0)


def test_negative_seed_size_raises():
    with pytest.raises(ValueError):
        load_dataset(seed_size=-1, val_size=50, test_size=100, random_state=0)
