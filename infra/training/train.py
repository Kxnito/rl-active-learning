"""SageMaker training entry point for the infrastructure smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments passed by SageMaker."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def find_npz(input_directory: Path) -> Path:
    """Find the prepared dataset file in the SageMaker input channel."""

    npz_files = sorted(input_directory.rglob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(
            f"No .npz dataset found under {input_directory}"
        )

    return npz_files[0]


def main() -> None:
    """Train a smoke-test classifier using Person A's dataset splits."""

    args = parse_args()

    training_directory = Path(
        os.environ.get(
            "SM_CHANNEL_TRAINING",
            "/opt/ml/input/data/training",
        )
    )

    model_directory = Path(
        os.environ.get(
            "SM_MODEL_DIR",
            "/opt/ml/model",
        )
    )

    output_directory = Path(
        os.environ.get(
            "SM_OUTPUT_DATA_DIR",
            "/opt/ml/output/data",
        )
    )

    model_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_path = find_npz(training_directory)

    print(f"Loading dataset from: {dataset_path}")

    dataset = np.load(dataset_path)

    seed_x = dataset["seed_X"]
    seed_y = dataset["seed_y"]

    pool_x = dataset["pool_X"]
    pool_y = dataset["pool_y"]

    val_x = dataset["val_X"]
    val_y = dataset["val_y"]

    test_x = dataset["test_X"]
    test_y = dataset["test_y"]

    print(f"Seed shape: {seed_x.shape}")
    print(f"Pool shape: {pool_x.shape}")
    print(f"Validation shape: {val_x.shape}")
    print(f"Test shape: {test_x.shape}")

    model = LogisticRegression(
        max_iter=2000,
        random_state=args.random_state,
    )

    model.fit(seed_x, seed_y)

    predictions = model.predict(val_x)
    validation_accuracy = accuracy_score(
        val_y,
        predictions,
    )

    metrics = {
        "algorithm": "logistic-regression-smoke-test",
        "validation_accuracy": float(validation_accuracy),
        "seed_rows": int(len(seed_x)),
        "pool_rows": int(len(pool_x)),
        "validation_rows": int(len(val_x)),
        "test_rows": int(len(test_x)),
        "random_state": args.random_state,
    }

    model_path = model_directory / "model.joblib"
    metrics_path = output_directory / "metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"validation_accuracy="
        f"{validation_accuracy:.6f}"
    )

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()