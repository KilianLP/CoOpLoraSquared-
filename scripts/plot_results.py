"""
Plot accuracy results from experiment CSV files produced by main.py.

Features:
 - Recursively scans a results root for CSVs whose filenames contain a pattern (e.g., "MCE").
 - Groups files by experiment label substrings (exp2/exp4/exp8/exp16 by default).
 - Pulls a chosen metric column (acc_test_new or acc_test_base) for each dataset.
 - Builds grouped bar plots per dataset and mean bars across methods.
 - Optionally overlays baseline methods provided via CLI.

Usage example:
    python scripts/plot_results.py \
        --root ./results \
        --pattern MCE \
        --experiments exp2 exp4 exp8 exp16 \
        --metric acc_test_new \
        --outdir plots

If you want baselines:
    python scripts/plot_results.py ... \
        --baseline "CLIP:36.29,77.80,64.05,59.90,97.26,77.50,94.00" \
        --baseline "LoRA:26.03,68.61,62.50,62.84,95.71,72.74,93.05"

Datasets default to a standard order; override with --datasets "dtd,eurosat,...".
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASETS = [
    "fgvc",
    "oxford_flowers",
    "eurosat",
    "dtd",
    "oxford_pets",
    "ucf101",
    "caltech101",
    "food101",
    "imagenet",
    "stanford_cars",
    "sun397",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="./results", help="Root directory containing result CSVs.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="MCE",
        help="Filename substring to select experiment CSVs (case-insensitive).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["exp2", "exp4", "exp8", "exp16"],
        help="Experiment label substrings used to group CSVs.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset order. Defaults to a preset list.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="acc_test_new",
        choices=["acc_test_new", "acc_test_base", "acc_test"],
        help="Metric column to read from each CSV.",
    )
    parser.add_argument("--outdir", type=str, default="plots", help="Where to write plots and summary CSV.")
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help=(
            "Optional baseline in the form 'Label:v1,v2,...' matching dataset order. "
            "You can pass this flag multiple times."
        ),
    )
    return parser.parse_args()


def parse_baselines(raw: Sequence[str], n_datasets: int) -> Dict[str, List[float]]:
    baselines: Dict[str, List[float]] = {}
    for item in raw:
        if ":" not in item:
            raise ValueError("Baseline must look like 'Name:1,2,3'")
        name, values = item.split(":", 1)
        vals = [float(x) for x in values.split(",")]
        if len(vals) != n_datasets:
            raise ValueError(f"Baseline {name} length {len(vals)} does not match dataset count {n_datasets}")
        baselines[name.strip()] = vals
    return baselines


def find_csvs(root: str, pattern: str) -> List[str]:
    files = glob.glob(os.path.join(root, "**", f"*{pattern}*.csv"), recursive=True)
    print(f"Found {len(files)} CSV files matching pattern '{pattern}'.")
    return files


def group_by_experiment(files: Sequence[str], experiments: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {exp: [] for exp in experiments}
    for f in files:
        fname = os.path.basename(f).lower()
        for exp in experiments:
            if exp.lower() in fname:
                groups[exp].append(f)
                break
    for exp, flist in groups.items():
        print(f"{exp}: {len(flist)} files")
    return groups


def read_metric(path: str, metric: str) -> float | None:
    try:
        with open(path) as f:
            row = next(csv.DictReader(f))
        return float(row[metric])
    except Exception as e:  # noqa: broad-except
        print(f"Error reading {metric} from {path}: {e}")
        return None


def values_for_datasets(files: Sequence[str], datasets: Sequence[str], metric: str) -> List[float | None]:
    vals: List[float | None] = []
    lower_files = [(p, p.lower()) for p in files]
    for ds in datasets:
        ds_l = ds.lower()
        match = [p for p, l in lower_files if ds_l in l]
        if not match:
            vals.append(None)
        else:
            vals.append(read_metric(match[0], metric))
    return vals


def nan_array(vals: Sequence[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in vals], dtype=float)


def plot_grouped(datasets: Sequence[str], methods: Sequence[str], matrix: np.ndarray, outdir: Path, metric: str):
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    plt.figure(figsize=(max(10, len(datasets) * 1.2), 6))
    for i, method in enumerate(methods):
        plt.bar(x + i * width, matrix[:, i], width, label=method)
    plt.xticks(x + (len(methods) - 1) * width / 2, datasets, rotation=40, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} per dataset")
    plt.legend()
    plt.tight_layout()
    outfile = outdir / f"{metric}_per_dataset.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {outfile}")


def plot_means(methods: Sequence[str], matrix: np.ndarray, outdir: Path, metric: str):
    means = np.nanmean(matrix, axis=0)
    plt.figure(figsize=(8, 5))
    plt.bar(methods, means)
    plt.ylabel(metric)
    plt.title(f"Mean {metric} across datasets")
    plt.xticks(rotation=25)
    plt.tight_layout()
    outfile = outdir / f"{metric}_means.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {outfile}")
    return means


def write_summary(datasets: Sequence[str], methods: Sequence[str], matrix: np.ndarray, means: np.ndarray, outdir: Path, metric: str):
    summary_path = outdir / f"{metric}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", *methods])
        for i, ds in enumerate(datasets):
            writer.writerow([ds, *matrix[i]])
        writer.writerow(["mean", *means])
    print(f"Saved {summary_path}")


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else DEFAULT_DATASETS

    files = find_csvs(args.root, args.pattern)
    groups = group_by_experiment(files, args.experiments)

    exp_vals: Dict[str, List[float | None]] = {}
    for exp, flist in groups.items():
        exp_vals[exp] = values_for_datasets(flist, datasets, args.metric)

    baselines = parse_baselines(args.baseline, len(datasets)) if args.baseline else {}

    methods = list(baselines.keys()) + list(exp_vals.keys())
    columns: List[np.ndarray] = [nan_array(baselines[name]) for name in baselines]
    columns.extend([nan_array(vals) for vals in exp_vals.values()])

    if not columns:
        raise SystemExit("No data to plot.")

    matrix = np.vstack(columns).T  # shape: datasets x methods
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_grouped(datasets, methods, matrix, outdir, args.metric)
    means = plot_means(methods, matrix, outdir, args.metric)
    write_summary(datasets, methods, matrix, means, outdir, args.metric)


if __name__ == "__main__":
    main()
