"""Load and validate active-learning experiment result files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "run_id",
    "method",
    "dataset",
    "seed",
    "step",
    "labels_used",
    "val_accuracy",
    "test_accuracy",
    "reward",
}

NUMERIC_COLUMNS = {
    "seed",
    "step",
    "labels_used",
    "val_accuracy",
    "test_accuracy",
    "reward",
}


def validate_results(
    dataframe: pd.DataFrame,
    source: str = "results",
) -> None:
    """Validate the result schema and basic value constraints."""

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{source} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(f"{source} contains no result rows.")

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise ValueError(
                f"Column '{column}' in {source} must be numeric."
            )

    if dataframe["run_id"].isna().any():
        raise ValueError(f"{source} contains missing run_id values.")

    if dataframe["method"].isna().any():
        raise ValueError(f"{source} contains missing method values.")

    if (dataframe["labels_used"] < 0).any():
        raise ValueError(
            f"{source} contains negative labels_used values."
        )

    for column in ("val_accuracy", "test_accuracy"):
        non_missing = dataframe[column].dropna()

        if ((non_missing < 0) | (non_missing > 1)).any():
            raise ValueError(
                f"Column '{column}' in {source} must be between 0 and 1."
            )


def load_result_file(path: str | Path) -> pd.DataFrame:
    """Load and validate one result CSV file."""

    result_path = Path(path)

    if not result_path.is_file():
        raise FileNotFoundError(
            f"Result file not found: {result_path}"
        )

    dataframe = pd.read_csv(result_path)

    validate_results(
        dataframe,
        source=str(result_path),
    )

    return dataframe


def load_result_directory(
    directory: str | Path,
) -> pd.DataFrame:
    """Load and combine every CSV file in a result directory."""

    result_directory = Path(directory)

    if not result_directory.is_dir():
        raise FileNotFoundError(
            f"Result directory not found: {result_directory}"
        )

    result_files = sorted(result_directory.rglob("*.csv"))

    if not result_files:
        raise FileNotFoundError(
            f"No CSV result files found under {result_directory}"
        )

    dataframes = [
        load_result_file(path)
        for path in result_files
    ]

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    validate_results(
        combined,
        source=str(result_directory),
    )

    return combined