"""Central AWS configuration for the project."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AWSConfig:
    """AWS resources used by the active-learning project."""

    region: str
    bucket: str
    project_prefix: str
    sagemaker_role_arn: str

    @classmethod
    def from_environment(cls) -> "AWSConfig":
        """Load configuration from environment variables."""

        region = os.getenv("AWS_REGION")
        bucket = os.getenv("S3_BUCKET")
        role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
        project_prefix = os.getenv(
            "PROJECT_PREFIX",
            "rl-active-learning",
        )

        missing = []

        if not region:
            missing.append("AWS_REGION")

        if not bucket:
            missing.append("S3_BUCKET")

        if not role_arn:
            missing.append("SAGEMAKER_ROLE_ARN")

        if missing:
            raise ValueError(
                "Missing required environment variables: "
                + ", ".join(missing)
            )

        return cls(
            region=region,
            bucket=bucket,
            project_prefix=project_prefix,
            sagemaker_role_arn=role_arn,
        )
