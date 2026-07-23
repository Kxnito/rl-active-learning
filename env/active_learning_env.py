"""
Custom Gymnasium environment for the active learning problem. State/action/
reward framing is defined in project-context.md Section 1:

  - state:   current student model's uncertainty across the pool,
             labels used so far, class balance
  - action:  select which unlabeled pool sample to query next
  - reward:  val accuracy after retraining, minus val accuracy before
  - episode: starts from the seed set, ends at the labeling budget

Why a fixed-size masked action space, not a shrinking one:
Stable-Baselines3's action spaces are fixed for the lifetime of training,
but the unlabeled pool shrinks by one every step as labels get revealed.
The fix is a Discrete(pool_capacity) action space over the *original* pool
size, where already-revealed indices are marked invalid via action_masks().
This requires MaskablePPO from sb3-contrib (not plain SB3 DQN/PPO, which
don't support action masking) — see agent/train.py.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data.dataset import DatasetSplits
from data.oracle import Oracle


class ActiveLearningEnv(gym.Env):
    def __init__(self, splits: DatasetSplits, budget: int):
        super().__init__()
        self.splits = splits
        self.budget = budget
        pool_capacity = len(splits.pool_X)

        self.action_space = spaces.Discrete(pool_capacity)
        # TODO(Person B): define the real observation space once _get_obs()'s
        # feature vector is decided (e.g. Box over [mean pool uncertainty,
        # labels_used / budget, class balance, ...]).
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.oracle: Optional[Oracle] = None
        self.student_model = None

    def reset(self, *, seed=None, options=None):
        """
        TODO(Person B): re-create a fresh Oracle over splits.pool_y, retrain
        the student model on just the seed set, and return the initial
        observation via _get_obs().
        """
        super().reset(seed=seed)
        raise NotImplementedError

    def step(self, action: int):
        """
        TODO(Person B):
          1. reveal(action) via self.oracle
          2. retrain the student model on seed + all revealed labels
          3. compute reward via _compute_reward()
          4. terminated = self.oracle.num_revealed >= self.budget
          5. return (obs, reward, terminated, truncated=False, info)
        """
        raise NotImplementedError

    def action_masks(self) -> np.ndarray:
        """
        Required by MaskablePPO. True = valid action (not yet revealed).
        TODO(Person B): build from self.oracle.is_revealed(i) for each pool
        index.
        """
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        """
        TODO(Person B): build the state vector — student model's
        uncertainty across the remaining pool, labels_used / budget, class
        balance among revealed labels so far.
        """
        raise NotImplementedError

    def _compute_reward(self, val_accuracy_before: float, val_accuracy_after: float) -> float:
        """reward_t = val_accuracy(model_t) - val_accuracy(model_{t-1}) — see project-context.md Section 8."""
        return val_accuracy_after - val_accuracy_before
