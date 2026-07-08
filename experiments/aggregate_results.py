"""Aggregate multi-seed experiment summaries into paper-style tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def mean_std(series):
    return f"{series.mean():.4f} +/- {series.std(ddof=0):.4f}"


def aggregate(summary_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(summary_csv)
    group_cols = ["method", "holdout_sensor", "label_fraction", "ablation"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n"] = len(group)
        for metric in ["test_macro_f1", "test_balanced_accuracy", "test_accuracy", "trainable_params", "elapsed_sec"]:
            row[metric] = mean_std(group[metric])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(group_cols)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(out.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.summary_csv.with_name("aggregate.csv")
    aggregate(args.summary_csv, output)


if __name__ == "__main__":
    main()
