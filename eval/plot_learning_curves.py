"""Plot active-learning accuracy against labeling budget."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eval.load_results import load_result_directory


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot average validation accuracy versus labels used."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing result CSV files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/plots/learning_curves.png"
        ),
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset name to plot.",
    )

    return parser.parse_args()


def aggregate_curves(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate accuracy across seeds for each method and budget."""

    return (
        dataframe.groupby(
            ["method", "labels_used"],
            as_index=False,
        )
        .agg(
            mean_accuracy=("val_accuracy", "mean"),
            std_accuracy=("val_accuracy", "std"),
            runs=("run_id", "nunique"),
        )
        .sort_values(
            ["method", "labels_used"]
        )
    )


def main() -> None:
    """Load results and save a learning-curve plot."""

    args = parse_args()

    dataframe = load_result_directory(
        args.input_dir
    )

    if args.dataset is not None:
        dataframe = dataframe[
            dataframe["dataset"] == args.dataset
        ]

        if dataframe.empty:
            raise ValueError(
                f"No results found for dataset "
                f"'{args.dataset}'."
            )

    aggregate = aggregate_curves(dataframe)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for method, method_data in aggregate.groupby(
        "method"
    ):
        ordered = method_data.sort_values(
            "labels_used"
        )

        x_values = ordered[
            "labels_used"
        ].to_numpy()

        mean_values = ordered[
            "mean_accuracy"
        ].to_numpy()

        std_values = ordered[
            "std_accuracy"
        ].fillna(0).to_numpy()

        axis.plot(
            x_values,
            mean_values,
            marker="o",
            label=method,
        )

        axis.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
        )

    dataset_label = (
        args.dataset
        if args.dataset is not None
        else "All datasets"
    )

    axis.set_title(
        f"Active Learning Performance — {dataset_label}"
    )
    axis.set_xlabel("Labels Used")
    axis.set_ylabel("Validation Accuracy")
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Method")

    figure.tight_layout()
    figure.savefig(
        args.output,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()