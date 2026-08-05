import numpy as np

from data.dataset import load_dataset

TOTAL_SAMPLES = 569  # Breast Cancer Wisconsin
OVERALL_POSITIVE_RATE = 357 / TOTAL_SAMPLES  # class balance in the full dataset


def test_split_sizes_match_requested_sizes():
    splits = load_dataset(seed_size=20, val_size=50, test_size=100, random_state=0)
    assert splits.seed_X.shape[0] == 20
    assert splits.val_X.shape[0] == 50
    assert splits.test_X.shape[0] == 100
    assert splits.pool_X.shape[0] == TOTAL_SAMPLES - 20 - 50 - 100


def test_splits_sum_to_total_dataset_size():
    splits = load_dataset(seed_size=20, val_size=50, test_size=100, random_state=0)
    total = (
        splits.seed_X.shape[0]
        + splits.pool_X.shape[0]
        + splits.val_X.shape[0]
        + splits.test_X.shape[0]
    )
    assert total == TOTAL_SAMPLES


def test_X_and_y_lengths_match_within_each_split():
    splits = load_dataset(seed_size=20, val_size=50, test_size=100, random_state=0)
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
