import numpy as np
import pytest

from data.oracle import Oracle

POOL_Y = np.array([0, 1, 0, 1, 1])


@pytest.fixture
def oracle():
    return Oracle(POOL_Y.copy())


def test_reveal_returns_the_true_label(oracle):
    assert oracle.reveal(2) == POOL_Y[2]
    assert oracle.reveal(3) == POOL_Y[3]


def test_reveal_marks_the_index_as_revealed(oracle):
    assert not oracle.is_revealed(1)
    oracle.reveal(1)
    assert oracle.is_revealed(1)


def test_revealing_same_index_twice_raises(oracle):
    oracle.reveal(0)
    with pytest.raises(ValueError):
        oracle.reveal(0)


def test_out_of_range_index_raises(oracle):
    with pytest.raises(IndexError):
        oracle.reveal(len(POOL_Y))  # one past the end
    with pytest.raises(IndexError):
        oracle.reveal(-1)  # negative indices are rejected, not Python-style wrapped


def test_is_revealed_rejects_out_of_range_same_as_reveal(oracle):
    """is_revealed() and reveal() must agree on what counts as a valid
    index — a negative index should not silently wrap to the last element."""
    with pytest.raises(IndexError):
        oracle.is_revealed(len(POOL_Y))
    with pytest.raises(IndexError):
        oracle.is_revealed(-1)


def test_num_revealed_starts_at_zero(oracle):
    assert oracle.num_revealed == 0


def test_num_revealed_increments_with_each_reveal(oracle):
    oracle.reveal(0)
    assert oracle.num_revealed == 1
    oracle.reveal(4)
    assert oracle.num_revealed == 2


def test_num_revealed_unaffected_by_failed_reveal(oracle):
    """A rejected reveal() call shouldn't leave the oracle in a half-updated state."""
    oracle.reveal(0)
    with pytest.raises(ValueError):
        oracle.reveal(0)
    assert oracle.num_revealed == 1
