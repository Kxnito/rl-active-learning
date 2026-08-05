"""Upload a dataset file to the project S3 bucket."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from infra.config import AWSConfig
from infra.s3_utils import S3Storage


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Upload a dataset file to project S3 storage."
    )

    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Local dataset file to upload.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, such as breast-cancer.",
    )

    parser.add_argument(
        "--stage",
        choices=["raw", "processed"],
        default="processed",
        help="Dataset processing stage.",
    )

    return parser.parse_args()


def main() -> None:
    """Upload the selected dataset file."""

    load_dotenv()
    args = parse_args()
    config = AWSConfig.from_environment()

    storage = S3Storage(
        bucket=config.bucket,
        region=config.region,
    )

    s3_key = (
        f"{config.project_prefix}/datasets/"
        f"{args.dataset}/{args.stage}/{args.file.name}"
    )

    uri = storage.upload_file(
        local_path=args.file,
        s3_key=s3_key,
    )

    print(f"Uploaded dataset: {uri}")


if __name__ == "__main__":
    main()