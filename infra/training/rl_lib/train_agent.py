"""
VENDORED from agent/train.py's train_agent() — see dataset.py in this
package for why. Keep this in sync with the real file; nothing here should
diverge from it.
"""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from .active_learning_env import ActiveLearningEnv
from .dataset import DatasetSplits


def mask_fn(env: ActiveLearningEnv):
    return env.action_masks()


def train_agent(splits: DatasetSplits, budget: int, total_timesteps: int, verbose: int = 1) -> MaskablePPO:
    env = ActiveLearningEnv(splits, budget=budget)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO("MlpPolicy", env, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)
    return model
