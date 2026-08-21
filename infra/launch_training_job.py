"""Launch a Scikit-learn training job on Amazon SageMaker."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn

from infra.config import AWSConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Launch the active-learning SageMaker training job."
    )

    parser.add_argument(
        "--input-s3-uri",
        required=True,
        help="S3 URI containing the prepared .npz dataset.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for training stochasticity (random sampling's "
        "choices, MaskablePPO). Does not reseed the data split itself — "
        "that's fixed at upload time in the .npz file.",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=50,
        help="Labeling budget per method (project-context.md Section 8).",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000,
        help="MaskablePPO training timesteps.",
    )

    parser.add_argument(
        "--dataset-name",
        default="breast_cancer",
        help="Dataset label recorded in the result CSV's dataset column.",
    )

    parser.add_argument(
        "--instance-type",
        default="ml.m5.large",
        help="SageMaker training instance type.",
    )

    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the training job to finish and stream logs.",
    )

    return parser.parse_args()


def main() -> None:
    """Configure and launch the SageMaker training job."""

    load_dotenv()

    args = parse_args()
    config = AWSConfig.from_environment()

    boto_session = boto3.Session(
        region_name=config.region,
    )

    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"active-learning-experiment-{timestamp}"

    output_path = (
        f"s3://{config.bucket}/"
        f"{config.project_prefix}/training-output"
    )

    training_source_dir = (
        Path(__file__).parent / "training"
    )

    if not training_source_dir.is_dir():
        raise FileNotFoundError(
            f"Training source directory not found: "
            f"{training_source_dir}"
        )

    training_script = (
        training_source_dir / "train.py"
    )

    if not training_script.is_file():
        raise FileNotFoundError(
            f"Training script not found: "
            f"{training_script}"
        )

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(training_source_dir),
        role=config.sagemaker_role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version="1.4-2",
        py_version="py3",
        output_path=output_path,
        base_job_name="active-learning-experiment",
        sagemaker_session=sagemaker_session,

        # New SageMaker accounts cannot use SageMaker Debugger.
        # Explicitly disable Debugger and profiling.
        debugger_hook_config=False,
        disable_profiler=True,

        hyperparameters={
            "random-state": args.random_state,
            "budget": args.budget,
            "total-timesteps": args.total_timesteps,
            "dataset-name": args.dataset_name,
            },
        metric_definitions=[
            {"Name": "random:val_accuracy", "Regex": r"random_final_val_accuracy=([0-9.]+)"},
            {"Name": "uncertainty:val_accuracy", "Regex": r"uncertainty_final_val_accuracy=([0-9.]+)"},
            {"Name": "rl:val_accuracy", "Regex": r"rl_final_val_accuracy=([0-9.]+)"},
            {"Name": "rl:test_accuracy", "Regex": r"rl_final_test_accuracy=([0-9.]+)"},
        ],
        # RL training (MaskablePPO) plus installing torch/sb3-contrib in
        # the container takes noticeably longer than the old smoke test's
        # single LogisticRegression fit — still comfortably under an hour
        # locally (budget=50/timesteps=10_000 ran in well under a minute
        # once dependencies were installed), but leaving headroom.
        max_run=3600,
    )

    training_input = TrainingInput(
        s3_data=args.input_s3_uri,
        content_type="application/octet-stream",
        input_mode="File",
    )

    print(f"Starting training job: {job_name}")
    print(f"Region: {config.region}")
    print(f"Input: {args.input_s3_uri}")
    print(f"Output: {output_path}")
    print(f"Instance: {args.instance_type}")
    print(f"Random state: {args.random_state}")

    estimator.fit(
        inputs={
            "training": training_input,
        },
        job_name=job_name,
        wait=args.wait,
        logs=args.wait,
    )

    print(f"Training job submitted: {job_name}")

    if not args.wait:
        print(
            "\nCheck status with:\n"
            "aws sagemaker describe-training-job "
            f"--training-job-name {job_name} "
            f"--region {config.region}"
        )


if __name__ == "__main__":
    main()