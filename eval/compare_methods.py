"""Generate summary tables comparing active-learning methods."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.load_results import load_result_directory
from eval.metrics import summarize_methods, summarize_runs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compare active-learning experiment methods."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
    )

    return parser.parse_args()


def main() -> None:
    """Load experiment results and write summary CSV files."""

    args = parse_args()

    dataframe = load_result_directory(
        args.input_dir
    )

    run_summary = summarize_runs(dataframe)
    method_summary = summarize_methods(run_summary)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_path = args.output_dir / "run_summary.csv"
    method_path = (
        args.output_dir / "method_summary.csv"
    )

    run_summary.to_csv(
        run_path,
        index=False,
    )

    method_summary.to_csv(
        method_path,
        index=False,
    )

    print("\nMethod comparison:")
    print(method_summary.to_string(index=False))

    print(f"\nSaved: {run_path}")
    print(f"Saved: {method_path}")


if __name__ == "__main__":
    main()