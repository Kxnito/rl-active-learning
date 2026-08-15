import numpy as np
import pytest

from data.dataset import load_dataset
from env.active_learning_env import ActiveLearningEnv

SEED_SIZE, VAL_SIZE, TEST_SIZE, BUDGET = 20, 50, 100, 50


@pytest.fixture
def splits():
    return load_dataset(seed_size=SEED_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=0)


@pytest.fixture
def env(splits):
    return ActiveLearningEnv(splits, budget=BUDGET)


# --- reset() ---

def test_reset_returns_correct_shape_and_dtype(env):
    obs, info = env.reset()
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_reset_obs_within_observation_space(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_reset_labels_used_frac_is_zero(env):
    obs, _ = env.reset()
    assert obs[1] == 0.0


def test_reset_class_balance_matches_seed(env, splits):
    obs, _ = env.reset()
    assert obs[2] == pytest.approx(splits.seed_y.mean())


def test_reset_clears_state_from_previous_episode(env):
    env.reset()
    env.step(0)
    env.step(1)
    assert env.oracle.num_revealed == 2

    obs, _ = env.reset()
    assert env.oracle.num_revealed == 0
    assert obs[1] == 0.0


# --- step() ---

def test_step_returns_five_tuple(env):
    env.reset()
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (3,)
    assert isinstance(reward, (int, float, np.floating))
    assert terminated in (True, False)
    assert truncated is False
    assert isinstance(info, dict)


def test_step_increases_labels_used_frac(env):
    env.reset()
    obs, *_ = env.step(0)
    assert obs[1] == pytest.approx(1 / BUDGET)


def test_step_reward_matches_accuracy_delta(env):
    env.reset()
    val_acc_before = env.student_model.score(env._scaler.transform(env.val_X), env.val_y)
    _, reward, *_ = env.step(0)
    val_acc_after = env.student_model.score(env._scaler.transform(env.val_X), env.val_y)
    assert reward == pytest.approx(val_acc_after - val_acc_before)


def test_terminated_true_exactly_at_budget(splits):
    small_budget = 3
    env = ActiveLearningEnv(splits, budget=small_budget)
    env.reset()
    for i in range(small_budget - 1):
        _, _, terminated, _, _ = env.step(i)
        assert not terminated
    _, _, terminated, _, _ = env.step(small_budget - 1)
    assert terminated


# --- action_masks() ---

def test_action_masks_starts_all_true(env):
    env.reset()
    assert env.action_masks().all()


def test_action_masks_shape_matches_pool_size(env, splits):
    env.reset()
    assert env.action_masks().shape == (len(splits.pool_X),)


def test_action_masks_flips_after_step(env):
    env.reset()
    env.step(5)
    mask = env.action_masks()
    assert mask[5] == False
    assert mask.sum() == len(mask) - 1


def test_multiple_steps_reduce_valid_actions(env):
    env.reset()
    env.step(0)
    env.step(1)
    env.step(2)
    mask = env.action_masks()
    assert mask.sum() == len(mask) - 3
    assert not mask[0] and not mask[1] and not mask[2]


# --- edge cases ---

def test_step_on_already_revealed_action_raises(env):
    env.reset()
    env.step(3)
    with pytest.raises(ValueError):
        env.step(3)


def test_step_on_out_of_range_action_raises(env, splits):
    env.reset()
    with pytest.raises(IndexError):
        env.step(len(splits.pool_X))  # one past the end
    with pytest.raises(IndexError):
        env.step(-1)