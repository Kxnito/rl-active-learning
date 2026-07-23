"""
Trains the RL agent (the policy that decides which pool sample to query
next) against ActiveLearningEnv. Uses MaskablePPO from sb3-contrib, not
plain SB3 DQN/PPO — see env/active_learning_env.py's module docstring for
why action masking is required here.
"""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from data.dataset import load_dataset
from env.active_learning_env import ActiveLearningEnv

# TODO(Person B): move these to config/CLI args once they're being tuned
SEED_SIZE = 20
VAL_SIZE = 100
TEST_SIZE = 100
BUDGET = 50  # project-context.md Section 8
TOTAL_TIMESTEPS = 10_000  # placeholder — tune once an episode's step count is known


def mask_fn(env: ActiveLearningEnv):
    return env.action_masks()


def main():
    splits = load_dataset(seed_size=SEED_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE)
    env = ActiveLearningEnv(splits, budget=BUDGET)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # TODO(Person B): save to a path under agent/ or checkpoints/ (gitignored —
    # real checkpoints belong in S3, per AGENTS.md conventions)
    model.save("agent/checkpoint_placeholder")


if __name__ == "__main__":
    main()
