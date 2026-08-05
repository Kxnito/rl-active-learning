"""Download experiment result CSV files from Amazon S3."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from infra.config import AWSConfig
from infra.s3_utils import S3Storage


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download active-learning result CSV files from S3."
    )

    parser.add_argument(
        "--prefix",
        default="results/",
        help="Prefix relative to the project root.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/downloaded/results"),
    )

    return parser.parse_args()


def main() -> None:
    """Download all CSV result files under the requested S3 prefix."""

    load_dotenv()

    args = parse_args()
    config = AWSConfig.from_environment()

    storage = S3Storage(
        bucket=config.bucket,
        region=config.region,
    )

    full_prefix = (
        f"{config.project_prefix}/"
        f"{args.prefix.lstrip('/')}"
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

    for key in csv_keys:
        relative_path = key.removeprefix(
            full_prefix
        ).lstrip("/")

        destination = (
            args.output_dir / relative_path
        )

        storage.download_file(
            s3_key=key,
            local_path=destination,
        )

        print(f"Downloaded: {destination}")


if __name__ == "__main__":
    main()