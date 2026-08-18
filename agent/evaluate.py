"""
Runs a trained RL agent through one deterministic episode against
ActiveLearningEnv, producing the same (labels_used, val_accuracy) curve
shape as data/baseline.py's run_random_sampling()/run_uncertainty_sampling()
— so eval/ can compare all three methods on equal footing.
"""

from typing import List, Tuple

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from data.dataset import DatasetSplits
from env.active_learning_env import ActiveLearningEnv


def mask_fn(env: ActiveLearningEnv):
    return env.action_masks()


def evaluate_agent(model: MaskablePPO, splits: DatasetSplits, budget: int) -> Tuple[List[Tuple[int, float]], float]:
    """Returns (curve, final_test_accuracy) — curve matches baseline.py's
    [(labels_used, val_accuracy), ...] shape; final_test_accuracy is scored
    once, after the episode ends, purely for reporting (never seen by the
    policy or folded into training/reward — same held-out rule as elsewhere)."""
    env = ActiveLearningEnv(splits, budget=budget)
    env = ActionMasker(env, mask_fn)

    # baseline.py counts labels_used as seed + queried (see its
    # run_random_sampling()/run_uncertainty_sampling()) — match that
    # convention exactly, not just Oracle.num_revealed (which only counts
    # pool queries), or the two curves would disagree about what
    # "labels_used" means and the comparison wouldn't be apples-to-apples.
    seed_size = len(splits.seed_X)

    obs, info = env.reset()
    # Starts empty, not with a seed-only point — matches baseline.py's
    # run_random_sampling()/run_uncertainty_sampling(), which only append
    # a point per label revealed, so the curves stay directly comparable.
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
