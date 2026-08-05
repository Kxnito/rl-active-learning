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
        help="S3 URI containing the training CSV file.",
    )

    parser.add_argument(
        "--target-column",
        default="target",
        help="Name of the target column in the CSV.",
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

    boto_session = boto3.Session(region_name=config.region)

    sagemaker_session = sagemaker.Session(
        boto_session=boto_session
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"active-learning-smoke-{timestamp}"

    output_path = (
        f"s3://{config.bucket}/"
        f"{config.project_prefix}/training-output"
    )

    training_source_dir = Path(__file__).parent / "training"

    if not training_source_dir.is_dir():
        raise FileNotFoundError(
            f"Training source directory not found: "
            f"{training_source_dir}"
        )

    training_script = training_source_dir / "train.py"

    if not training_script.is_file():
        raise FileNotFoundError(
            f"Training script not found: {training_script}"
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
        base_job_name="active-learning-smoke",
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "target-column": args.target_column,
            "random-state": 42,
            "test-size": 0.2,
        },
        metric_definitions=[
            {
                "Name": "validation:accuracy",
                "Regex": (
                    r"validation_accuracy=([0-9.]+)"
                ),
            }
        ],
        max_run=3600,
    )

    training_input = TrainingInput(
        s3_data=args.input_s3_uri,
        content_type="text/csv",
        input_mode="File",
    )

    print(f"Starting training job: {job_name}")
    print(f"Region: {config.region}")
    print(f"Input: {args.input_s3_uri}")
    print(f"Output: {output_path}")
    print(f"Instance: {args.instance_type}")

    estimator.fit(
        inputs={"training": training_input},
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