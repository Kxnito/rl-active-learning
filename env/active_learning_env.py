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
        """
        TODO(Person B): re-create a fresh Oracle over self._pool_y, retrain
        the student model on just the seed set, and return the initial
        observation via _get_obs().
        """
        super().reset(seed=seed)

        self.oracle = Oracle(self._pool_y)

        self._revealed_X = []
        self._revealed_y = [] # Initialize a list to store revealed labels

        self.student_model = LogisticRegression()  # Initialize the student model
        self.student_model.fit(self._scaler.transform(self.seed_X), self.seed_y)  # Train the student model

        return self._get_obs(), {}

    def step(self, action: int):
        """
        TODO(Person B):
          1. reveal(action) via self.oracle
          2. retrain the student model on seed + all revealed labels
          3. compute reward via _compute_reward()
          4. terminated = self.oracle.num_revealed >= self.budget
          5. return (obs, reward, terminated, truncated=False, info)
        """
        val_accuracy_before = self.student_model.score(self._scaler.transform(self.val_X), self.val_y)  # Get validation accuracy before revealing

        label = self.oracle.reveal(action)  # Reveal the label for the selected action
        self._revealed_X.append(self.pool_X[action])  # Store the revealed feature
        self._revealed_y.append(label)  # Store the revealed label

        train_X = np.vstack([self.seed_X] + self._revealed_X) if self._revealed_X else self.seed_X  # Combine seed and revealed features
        train_y = np.concatenate([self.seed_y, self._revealed_y]) if self._revealed_y else self.seed_y  # Combine seed and revealed labels
        self.student_model.fit(self._scaler.transform(train_X), train_y)  # Retrain the student model

        val_accuracy_after = self.student_model.score(self._scaler.transform(self.val_X), self.val_y)  # Get validation accuracy after revealing
        reward = self._compute_reward(val_accuracy_before, val_accuracy_after)  # Compute the reward based on validation accuracy change

        terminated = self.oracle.num_revealed >= self.budget  # Check if the labeling budget has been reached

        return self._get_obs(), reward, terminated, False, {}  # Return the observation, reward, termination status, and info

    
    def action_masks(self) -> np.ndarray:
        """
        Required by MaskablePPO. True = valid action (not yet revealed).
        TODO(Person B): build from self.oracle.is_revealed(i) for each pool
        index.
        """
        return np.array([not self.oracle.is_revealed(i) for i in range(len(self.pool_X))], dtype=bool)  # Create a mask indicating valid actions

    def _get_obs(self) -> np.ndarray:
        """
        TODO(Person B): build the state vector — student model's
        uncertainty across the remaining pool, labels_used / budget, class
        balance among revealed labels so far.
        """
        probs = self.student_model.predict_proba(self._scaler.transform(self.pool_X))  # Get predicted probabilities for the pool
        uncertainty = 1 - probs.max(axis=1).mean()  # Calculate uncertainty

        labels_used_frac = self.oracle.num_revealed / self.budget  # Calculate fraction of labels used

        all_revealed_y = np.concatenate([self.seed_y, self._revealed_y]) if self._revealed_y else self.seed_y  # Combine seed and revealed labels
        class_balance = all_revealed_y.mean() # Calculate class balance

        return np.array([uncertainty, labels_used_frac, class_balance], dtype=np.float32)  # Return the observation vector

    def _compute_reward(self, val_accuracy_before: float, val_accuracy_after: float) -> float:
        """reward_t = val_accuracy(model_t) - val_accuracy(model_{t-1}) — see project-context.md Section 8."""
        return val_accuracy_after - val_accuracy_before
