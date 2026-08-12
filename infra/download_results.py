"""Download SageMaker training outputs or experiment CSV results from S3."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from dotenv import load_dotenv

from infra.config import AWSConfig
from infra.s3_utils import S3Storage


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download active-learning results from Amazon S3."
    )

    parser.add_argument(
        "--job-name",
        help=(
            "SageMaker training job name. If provided, download "
            "that job's output.tar.gz and extract metrics.json."
        ),
    )

    parser.add_argument(
        "--prefix",
        default="results/",
        help=(
            "Prefix relative to the project root when downloading "
            "experiment CSV files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/downloaded"),
        help="Local directory for downloaded results.",
    )

    return parser.parse_args()


def download_job_output(
    storage: S3Storage,
    config: AWSConfig,
    job_name: str,
    output_dir: Path,
) -> None:
    """Download and extract a SageMaker job's output.tar.gz."""

    s3_key = (
        f"{config.project_prefix}/"
        f"training-output/"
        f"{job_name}/"
        "output/output.tar.gz"
    )

    job_output_dir = output_dir / job_name
    job_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path = (
        job_output_dir / "output.tar.gz"
    )

    print(
        f"Downloading "
        f"s3://{config.bucket}/{s3_key}"
    )

    storage.download_file(
        s3_key=s3_key,
        local_path=archive_path,
    )

    print(f"Downloaded: {archive_path}")

    with tarfile.open(
        archive_path,
        mode="r:gz",
    ) as archive:
        archive.extractall(
            path=job_output_dir,
            filter="data",
        )

    print(f"Extracted results to: {job_output_dir}")

    metrics_path = (
        job_output_dir / "metrics.json"
    )

    if metrics_path.is_file():
        print(f"Metrics available at: {metrics_path}")
    else:
        print(
            "Warning: metrics.json was not found "
            f"inside {archive_path}"
        )


def download_csv_results(
    storage: S3Storage,
    config: AWSConfig,
    prefix: str,
    output_dir: Path,
) -> None:
    """Download all experiment CSV files under an S3 prefix."""

    full_prefix = (
        f"{config.project_prefix}/"
        f"{prefix.lstrip('/')}"
    )

    keys = storage.list_objects(full_prefix)

    csv_keys = [
        key
        for key in keys
        if key.lower().endswith(".csv")
    ]

    if not csv_keys:
        print(
            f"No CSV result files found under "
            f"s3://{config.bucket}/{full_prefix}"
        )
        return

    csv_output_dir = (
        output_dir / "results"
    )

    for key in csv_keys:
        relative_path = key.removeprefix(
            full_prefix
        ).lstrip("/")

        destination = (
            csv_output_dir / relative_path
        )

        storage.download_file(
            s3_key=key,
            local_path=destination,
        )

        print(f"Downloaded: {destination}")


def main() -> None:
    """Download SageMaker job outputs or experiment result CSVs."""

    load_dotenv()

    args = parse_args()
    config = AWSConfig.from_environment()

    storage = S3Storage(
        bucket=config.bucket,
        region=config.region,
    )

    if args.job_name:
        download_job_output(
            storage=storage,
            config=config,
            job_name=args.job_name,
            output_dir=args.output_dir,
        )
    else:
        download_csv_results(
            storage=storage,
            config=config,
            prefix=args.prefix,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()