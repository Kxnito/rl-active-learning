from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from data.dataset import load_dataset
from infra.config import AWSConfig
from infra.s3_utils import S3Storage


def prepare_dataset(
    output_path: str = "outputs/breast_cancer_splits.npz",
    seed_size: int = 10,
    val_size: int = 100,
    test_size: int = 100,
    random_state: int = 42,
) -> Path:
    """Create reproducible active-learning dataset splits."""

    splits = load_dataset(
        seed_size=seed_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        seed_X=splits.seed_X,
        seed_y=splits.seed_y,
        pool_X=splits.pool_X,
        pool_y=splits.pool_y,
        val_X=splits.val_X,
        val_y=splits.val_y,
        test_X=splits.test_X,
        test_y=splits.test_y,
    )

    print(f"Saved dataset splits to {output_path}")

    return output_path


def main() -> None:
    """Prepare the dataset and upload it to project S3 storage."""

    load_dotenv()

    config = AWSConfig.from_environment()

    local_path = prepare_dataset()

    storage = S3Storage(
        bucket=config.bucket,
        region=config.region,
    )

    s3_key = (
        f"{config.project_prefix}/"
        "data/processed/"
        "breast_cancer_splits.npz"
    )

    uri = storage.upload_file(
        local_path=local_path,
        s3_key=s3_key,
    )

    print(f"Uploaded dataset to: {uri}")


if __name__ == "__main__":
    main()