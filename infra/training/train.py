"""SageMaker training entry point — runs random sampling, uncertainty
sampling, and the RL agent on the dataset uploaded by infra/upload_data.py,
and writes results in the same CSV schema eval/run_experiment.py produces
locally, so infra/download_results.py + eval/compare_methods.py /
eval/plot_learning_curves.py work unchanged regardless of whether the run
happened locally or on SageMaker.

Note on --random-state here vs. eval/run_experiment.py's --seed locally:
the dataset .npz is already split (fixed at upload time by
infra/upload_data.py), so this only reseeds *training stochasticity*
(random-sampling's choices, MaskablePPO's own randomness) — it does not
reseed the train/val/test split itself the way running load_dataset()
fresh per seed does locally. Multiple SageMaker runs with different
--random-state values are not identical to eval/run_experiment.py's
multi-seed sweep for this reason.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from rl_lib.baseline import run_random_sampling, run_uncertainty_sampling
from rl_lib.dataset import DatasetSplits
from rl_lib.evaluate_agent import evaluate_agent
from rl_lib.oracle import Oracle
from rl_lib.train_agent import train_agent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments passed by SageMaker."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--dataset-name", default="breast_cancer")

    return parser.parse_args()


def find_npz(input_directory: Path) -> Path:
    """Find the prepared dataset file in the SageMaker input channel."""

    npz_files = sorted(input_directory.rglob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(
            f"No .npz dataset found under {input_directory}"
        )

    return npz_files[0]


def load_splits(npz_path: Path) -> DatasetSplits:
    """Reconstructs DatasetSplits from the .npz infra/upload_data.py wrote."""

    dataset = np.load(npz_path)

    return DatasetSplits(
        seed_X=dataset["seed_X"],
        seed_y=dataset["seed_y"],
        pool_X=dataset["pool_X"],
        pool_y=dataset["pool_y"],
        val_X=dataset["val_X"],
        val_y=dataset["val_y"],
        test_X=dataset["test_X"],
        test_y=dataset["test_y"],
    )


def curve_to_rows(
    curve: List[Tuple[int, float]],
    method: str,
    dataset: str,
    seed: int,
    final_test_accuracy: Optional[float] = None,
) -> List[dict]:
    """Same schema as eval/run_experiment.py's curve_to_rows() — see that
    file for the full reasoning behind each field."""

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
    """Run all three active-learning methods and write one combined
    result CSV plus the trained RL model."""

    args = parse_args()

    training_directory = Path(
        os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    )
    model_directory = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_directory = Path(
        os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )

    model_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_path = find_npz(training_directory)
    print(f"Loading dataset from: {dataset_path}")
    splits = load_splits(dataset_path)

    print(f"Seed shape: {splits.seed_X.shape}")
    print(f"Pool shape: {splits.pool_X.shape}")
    print(f"Validation shape: {splits.val_X.shape}")
    print(f"Test shape: {splits.test_X.shape}")

    rows: List[dict] = []

    print("Running random sampling...")
    random_curve = run_random_sampling(splits, Oracle(splits.pool_y), args.budget)
    rows += curve_to_rows(random_curve, "random", args.dataset_name, args.random_state)

    print("Running uncertainty sampling...")
    uncertainty_curve = run_uncertainty_sampling(splits, Oracle(splits.pool_y), args.budget)
    rows += curve_to_rows(uncertainty_curve, "uncertainty", args.dataset_name, args.random_state)

    print("Training RL agent...")
    model = train_agent(splits, args.budget, args.total_timesteps, verbose=1)

    print("Evaluating RL agent...")
    rl_curve, rl_test_accuracy = evaluate_agent(model, splits, args.budget)
    rows += curve_to_rows(
        rl_curve, "rl", args.dataset_name, args.random_state, final_test_accuracy=rl_test_accuracy
    )

    results_path = output_directory / "results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Saved results to: {results_path}")

    model_path = model_directory / "rl_model.zip"
    model.save(str(model_path.with_suffix("")))
    print(f"Saved RL model to: {model_path}")

    final_random_accuracy = random_curve[-1][1] if random_curve else float("nan")
    final_uncertainty_accuracy = uncertainty_curve[-1][1] if uncertainty_curve else float("nan")
    final_rl_accuracy = rl_curve[-1][1] if rl_curve else float("nan")

    metrics = {
        "random_final_val_accuracy": float(final_random_accuracy),
        "uncertainty_final_val_accuracy": float(final_uncertainty_accuracy),
        "rl_final_val_accuracy": float(final_rl_accuracy),
        "rl_final_test_accuracy": float(rl_test_accuracy),
        "budget": args.budget,
        "total_timesteps": args.total_timesteps,
        "random_state": args.random_state,
    }

    metrics_path = output_directory / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Printed in SageMaker's own metric_definitions regex format (see
    # infra/launch_training_job.py) so CloudWatch picks these up per run.
    print(f"random_final_val_accuracy={final_random_accuracy:.6f}")
    print(f"uncertainty_final_val_accuracy={final_uncertainty_accuracy:.6f}")
    print(f"rl_final_val_accuracy={final_rl_accuracy:.6f}")
    print(f"rl_final_test_accuracy={rl_test_accuracy:.6f}")

    print(f"Saved metrics to: {metrics_path}")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
