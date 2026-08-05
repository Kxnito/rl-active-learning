"""Reusable Amazon S3 file operations."""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


class S3Storage:
    """Upload, download, and list project files in Amazon S3."""

    def __init__(self, bucket: str, region: str) -> None:
        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region)

    def upload_file(
        self,
        local_path: str | Path,
        s3_key: str,
    ) -> str:
        """Upload a local file and return its S3 URI."""

        path = Path(local_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            self.client.upload_file(
                Filename=str(path),
                Bucket=self.bucket,
                Key=s3_key,
            )
        except (ClientError, BotoCoreError) as error:
            raise RuntimeError(
                f"Failed to upload {path} to "
                f"s3://{self.bucket}/{s3_key}"
            ) from error

        return f"s3://{self.bucket}/{s3_key}"

    def download_file(
        self,
        s3_key: str,
        local_path: str | Path,
    ) -> Path:
        """Download an S3 object to a local file."""

        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.download_file(
                Bucket=self.bucket,
                Key=s3_key,
                Filename=str(destination),
            )
        except (ClientError, BotoCoreError) as error:
            raise RuntimeError(
                f"Failed to download "
                f"s3://{self.bucket}/{s3_key}"
            ) from error

        return destination

    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys under an S3 prefix."""

        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
        ):
            for item in page.get("Contents", []):
                keys.append(item["Key"])

        return keys

    def object_exists(self, s3_key: str) -> bool:
        """Return whether an S3 object exists."""

        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            return True
        except ClientError as error:
            error_code = error.response.get(
                "Error",
                {},
            ).get("Code")

            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False

            raise