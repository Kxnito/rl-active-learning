"""
VENDORED from env/active_learning_env.py — see dataset.py in this package
for why. Keep this in sync with the real file; nothing here should diverge
from it.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .dataset import DatasetSplits
from .oracle import Oracle


class ActiveLearningEnv(gym.Env):
    def __init__(self, splits: DatasetSplits, budget: int):
        super().__init__()
        self.seed_X, self.seed_y = splits.seed_X, splits.seed_y
        self.pool_X = splits.pool_X
        self.val_X, self.val_y = splits.val_X, splits.val_y
        self.test_X, self.test_y = splits.test_X, splits.test_y
        self._pool_y = splits.pool_y

        self._scaler = StandardScaler().fit(np.concatenate([self.seed_X, self.pool_X]))
        self._seed_X_scaled = self._scaler.transform(self.seed_X)
        self._pool_X_scaled = self._scaler.transform(self.pool_X)
        self._val_X_scaled = self._scaler.transform(self.val_X)

        self.budget = budget
        pool_capacity = len(self.pool_X)

        self.action_space = spaces.Discrete(pool_capacity)
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

        return self._get_obs(), {"val_accuracy": self._val_accuracy}

    def step(self, action: int):
        val_accuracy_before = self._val_accuracy

        label = self.oracle.reveal(action)
        self._revealed_X_scaled.append(self._pool_X_scaled[action])
        self._revealed_y.append(label)

        train_X = np.vstack([self._seed_X_scaled] + self._revealed_X_scaled)
        train_y = self._labels_so_far()
        self.student_model.fit(train_X, train_y)

        self._val_accuracy = self.student_model.score(self._val_X_scaled, self.val_y)
        reward = self._compute_reward(val_accuracy_before, self._val_accuracy)

        terminated = self.oracle.num_revealed >= self.budget

        return self._get_obs(), reward, terminated, False, {"val_accuracy": self._val_accuracy}

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
