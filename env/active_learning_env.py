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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.dataset import DatasetSplits
from data.oracle import Oracle


class ActiveLearningEnv(gym.Env):
    def __init__(self, splits: DatasetSplits, budget: int):
        super().__init__()
        # Store only the env-safe pieces as named attributes, not the whole
        # DatasetSplits object — that would keep splits.pool_y (the ground
        # truth labels) reachable from every method for the env's entire
        # lifetime. pool_y is needed once per episode, only to build a
        # fresh Oracle in reset() — everywhere else must go through
        # self.oracle.reveal()/is_revealed(), never self._pool_y directly.
        self.seed_X, self.seed_y = splits.seed_X, splits.seed_y
        self.pool_X = splits.pool_X
        self.val_X, self.val_y = splits.val_X, splits.val_y
        self.test_X, self.test_y = splits.test_X, splits.test_y
        self._pool_y = splits.pool_y

        # Fit once on the full feature pool (labels aren't involved, so this
        # isn't leaking anything an active learning setup wouldn't already
        # have — only pool_y is meant to stay hidden). Fitting this fresh on
        # just the 20-sample seed set each episode was unstable: at least one
        # of 30 features had near-zero variance in such a small sample,
        # producing scaled values 15+ std devs out on the wider pool and
        # overflowing the model's matmul.
        self._scaler = StandardScaler().fit(np.concatenate([self.seed_X, self.pool_X]))
        # seed_X/pool_X/val_X never change after this — transform once here
        # rather than re-transforming the same constant arrays every step().
        self._seed_X_scaled = self._scaler.transform(self.seed_X)
        self._pool_X_scaled = self._scaler.transform(self.pool_X)
        self._val_X_scaled = self._scaler.transform(self.val_X)

        self.budget = budget
        pool_capacity = len(self.pool_X)

        self.action_space = spaces.Discrete(pool_capacity)
        # TODO(Person B): define the real observation space once _get_obs()'s
        # feature vector is decided (e.g. Box over [mean pool uncertainty,
        # labels_used / budget, class balance, ...]).
        self.observation_space = spaces.Box(low=-0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.oracle: Optional[Oracle] = None
        self.student_model = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.oracle = Oracle(self._pool_y)
        self._revealed_X_scaled = []
        self._revealed_y = []

        self.student_model = LogisticRegression()
        self.student_model.fit(self._seed_X_scaled, self.seed_y)
        self._val_accuracy = self.student_model.score(self._val_X_scaled, self.val_y)

        return self._get_obs(), {}

    def step(self, action: int):
        val_accuracy_before = self._val_accuracy

        label = self.oracle.reveal(action)
        self._revealed_X_scaled.append(self._pool_X_scaled[action])
        self._revealed_y.append(label)

        train_X = np.vstack([self._seed_X_scaled] + self._revealed_X_scaled)
        train_y = self._labels_so_far()
        self.student_model.fit(train_X, train_y)

        # Cached so the next step()'s val_accuracy_before doesn't redo this
        # scoring pass on data that hasn't changed since this line ran.
        self._val_accuracy = self.student_model.score(self._val_X_scaled, self.val_y)
        reward = self._compute_reward(val_accuracy_before, self._val_accuracy)

        terminated = self.oracle.num_revealed >= self.budget

        return self._get_obs(), reward, terminated, False, {}

    def action_masks(self) -> np.ndarray:
        """Required by MaskablePPO. True = valid action (not yet revealed)."""
        return ~self.oracle.revealed_mask

    def _labels_so_far(self) -> np.ndarray:
        """seed_y plus every label revealed so far this episode."""
        if not self._revealed_y:
            return self.seed_y
        return np.concatenate([self.seed_y, self._revealed_y])

    def _get_obs(self) -> np.ndarray:
        probs = self.student_model.predict_proba(self._pool_X_scaled)
        uncertainty = 1 - probs.max(axis=1).mean()
        labels_used_frac = self.oracle.num_revealed / self.budget
        class_balance = self._labels_so_far().mean()
        return np.array([uncertainty, labels_used_frac, class_balance], dtype=np.float32)

    def _compute_reward(self, val_accuracy_before: float, val_accuracy_after: float) -> float:
        """reward_t = val_accuracy(model_t) - val_accuracy(model_{t-1}) — see project-context.md Section 8."""
        return val_accuracy_after - val_accuracy_before
