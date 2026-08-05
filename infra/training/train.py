"""SageMaker training entry point for the infrastructure smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments passed by SageMaker."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-column",
        type=str,
        default="target",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def find_csv(input_directory: Path) -> Path:
    """Find the first CSV file in the SageMaker input channel."""

    csv_files = sorted(input_directory.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found under {input_directory}"
        )

    return csv_files[0]


def main() -> None:
    """Train a simple classifier and save model artifacts and metrics."""

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

    dataset_path = find_csv(training_directory)
    print(f"Loading dataset from: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)

    if args.target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{args.target_column}' not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    features = dataframe.drop(columns=[args.target_column])
    target = dataframe[args.target_column]

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=args.random_state,
    )
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)

    metrics = {
        "algorithm": "logistic-regression-smoke-test",
        "accuracy": float(accuracy),
        "training_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "target_column": args.target_column,
        "random_state": args.random_state,
    }

    model_path = model_directory / "model.joblib"
    metrics_path = output_directory / "metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"validation_accuracy={accuracy:.6f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
