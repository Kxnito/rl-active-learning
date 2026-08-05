"""Metrics for comparing active-learning experiment runs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def learning_curve_auc(
    labels_used: pd.Series,
    accuracy: pd.Series,
) -> float:
    """Calculate area under an accuracy-versus-labels curve."""

    valid = pd.DataFrame(
        {
            "labels_used": labels_used,
            "accuracy": accuracy,
        }
    ).dropna()

    if len(valid) < 2:
        return float("nan")

    valid = valid.sort_values("labels_used")

    x_values = valid["labels_used"].to_numpy(dtype=float)
    y_values = valid["accuracy"].to_numpy(dtype=float)

    return float(np.trapezoid(y_values, x_values))


def normalized_learning_curve_auc(
    labels_used: pd.Series,
    accuracy: pd.Series,
) -> float:
    """Calculate AUC normalized by the label-budget range."""

    raw_auc = learning_curve_auc(
        labels_used,
        accuracy,
    )

    if np.isnan(raw_auc):
        return float("nan")

    minimum = float(labels_used.min())
    maximum = float(labels_used.max())
    label_range = maximum - minimum

    if label_range <= 0:
        return float("nan")

    return raw_auc / label_range


def summarize_runs(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row for each experiment run."""

    summaries: list[dict[str, object]] = []

    for run_id, group in dataframe.groupby("run_id"):
        ordered = group.sort_values(
            ["labels_used", "step"]
        )

        final_row = ordered.iloc[-1]

        summaries.append(
            {
                "run_id": run_id,
                "method": final_row["method"],
                "dataset": final_row["dataset"],
                "seed": int(final_row["seed"]),
                "initial_labels_used": int(
                    ordered.iloc[0]["labels_used"]
                ),
                "final_labels_used": int(
                    final_row["labels_used"]
                ),
                "initial_val_accuracy": float(
                    ordered.iloc[0]["val_accuracy"]
                ),
                "final_val_accuracy": float(
                    final_row["val_accuracy"]
                ),
                "final_test_accuracy": float(
                    final_row["test_accuracy"]
                ),
                "total_reward": float(
                    ordered["reward"].sum()
                ),
                "learning_curve_auc": learning_curve_auc(
                    ordered["labels_used"],
                    ordered["val_accuracy"],
                ),
                "normalized_auc": (
                    normalized_learning_curve_auc(
                        ordered["labels_used"],
                        ordered["val_accuracy"],
                    )
                ),
            }
        )

    return pd.DataFrame(summaries)


def summarize_methods(
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate run summaries by dataset and method."""

    return (
        run_summary.groupby(
            ["dataset", "method"],
            as_index=False,
        )
        .agg(
            runs=("run_id", "nunique"),
            mean_final_val_accuracy=(
                "final_val_accuracy",
                "mean",
            ),
            std_final_val_accuracy=(
                "final_val_accuracy",
                "std",
            ),
            mean_final_test_accuracy=(
                "final_test_accuracy",
                "mean",
            ),
            std_final_test_accuracy=(
                "final_test_accuracy",
                "std",
            ),
            mean_normalized_auc=(
                "normalized_auc",
                "mean",
            ),
            std_normalized_auc=(
                "normalized_auc",
                "std",
            ),
            mean_total_reward=(
                "total_reward",
                "mean",
            ),
        )
        .sort_values(
            ["dataset", "mean_normalized_auc"],
            ascending=[True, False],
        )
    )