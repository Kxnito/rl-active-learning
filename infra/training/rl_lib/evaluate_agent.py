"""
VENDORED from agent/evaluate.py — see dataset.py in this package for why.
Keep this in sync with the real file; nothing here should diverge from it.
"""

from typing import List, Tuple

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from .active_learning_env import ActiveLearningEnv
from .dataset import DatasetSplits


def mask_fn(env: ActiveLearningEnv):
    return env.action_masks()


def evaluate_agent(model: MaskablePPO, splits: DatasetSplits, budget: int) -> Tuple[List[Tuple[int, float]], float]:
    """Returns (curve, final_test_accuracy)."""
    env = ActiveLearningEnv(splits, budget=budget)
    env = ActionMasker(env, mask_fn)

    seed_size = len(splits.seed_X)

    obs, info = env.reset()
    curve: List[Tuple[int, float]] = []

    terminated = truncated = False
    while not (terminated or truncated):
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        curve.append((seed_size + env.unwrapped.oracle.num_revealed, info["val_accuracy"]))

    final_test_accuracy = env.unwrapped.student_model.score(
        env.unwrapped._scaler.transform(env.unwrapped.test_X), env.unwrapped.test_y
    )

    return curve, final_test_accuracy
