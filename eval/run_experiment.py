"""Runs random sampling, uncertainty sampling, and a trained RL agent on the
same dataset/seed/budget, writing results in the schema load_results.py
expects. This is the missing link between data/baseline.py + agent/ and
eval/'s plotting/comparison tools — nothing else in the repo currently
produces eval/-compatible result files from real (non-example) runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from agent.evaluate import evaluate_agent
from agent.train import train_agent
from data.baseline import run_random_sampling, run_uncertainty_sampling
from data.dataset import load_dataset
from data.oracle import Oracle


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Run random/uncertainty/RL active learning and write comparable result CSVs."
    )

    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-size", type=int, default=20)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("local-test/results"))

    return parser.parse_args()


def curve_to_rows(
    curve: List[Tuple[int, float]],
    method: str,
    dataset: str,
    seed: int,
    final_test_accuracy: Optional[float] = None,
) -> List[dict]:
    """Reshapes a (labels_used, val_accuracy) curve into load_results.py's
    required row schema. test_accuracy is left blank except on the final
    row — per metrics.py's own comment, it only needs to exist where the
    held-out test set was actually evaluated, not at every step. reward is
    the accuracy delta from the previous step, matching how
    ActiveLearningEnv._compute_reward() defines it — so it means the same
    thing for every method's curve, not just the RL agent's."""

    rows = []
    prev_val_accuracy = None

    for step, (labels_used, val_accuracy) in enumerate(curve):
        reward = 0.0 if prev_val_accuracy is None else val_accuracy - prev_val_accuracy
        prev_val_accuracy = val_accuracy
        is_last_row = step == len(curve) - 1

        rows.append(
            {
                "run_id": f"{method}-{seed}",
                "method": method,
                "dataset": dataset,
                "seed": seed,
                "step": step,
                "labels_used": labels_used,
                "val_accuracy": val_accuracy,
                "test_accuracy": final_test_accuracy if (is_last_row and final_test_accuracy is not None) else float("nan"),
                "reward": reward,
            }
        )

    return rows


def main() -> None:
    """Run all three methods and write one combined result CSV."""

    args = parse_args()

    splits = load_dataset(
        seed_size=args.seed_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    rows: List[dict] = []

    print("Running random sampling...")
    random_curve = run_random_sampling(splits, Oracle(splits.pool_y), args.budget)
    rows += curve_to_rows(random_curve, "random", args.dataset, args.seed)

    print("Running uncertainty sampling...")
    uncertainty_curve = run_uncertainty_sampling(splits, Oracle(splits.pool_y), args.budget)
    rows += curve_to_rows(uncertainty_curve, "uncertainty", args.dataset, args.seed)

    print("Training RL agent...")
    model = train_agent(splits, args.budget, args.total_timesteps, verbose=0)

    print("Evaluating RL agent...")
    rl_curve, rl_test_accuracy = evaluate_agent(model, splits, args.budget)
    rows += curve_to_rows(rl_curve, "rl", args.dataset, args.seed, final_test_accuracy=rl_test_accuracy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"experiment-seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
