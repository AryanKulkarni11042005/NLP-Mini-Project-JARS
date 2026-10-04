"""Compute text metrics from the repository's persisted paired evaluation data.

The saved ``eval_sets.pkl`` in this repository contains only built-in Python
containers (dict/list/tuple/string) produced by 02_advance.ipynb. Do not point
this helper at arbitrary pickle files.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from .artifacts import write_csv_new
from .plots import plot_text_quality
from .robustness_metrics import summarize_replicates
from .text_metrics import evaluate_transformation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_saved_pairs(pickle_path: str | Path, output_dir: str | Path):
    with Path(pickle_path).open("rb") as stream:
        data = pickle.load(stream)

    required = {"X_test_hing", "NOISY", "NORM"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(f"Expected saved evaluation keys: {sorted(required)}")
    references = [str(value) for value in data["X_test_hing"]]
    rows = [evaluate_transformation(
        references, references, condition="Clean Hinglish", noise_level="none",
        normalization=False, seed="clean", dataset="Hindi emotion dataset",
    )]

    noisy, normalized = data["NOISY"], data["NORM"]
    keys = sorted(noisy, key=lambda item: (item[0], item[1]))
    if set(keys) != set(normalized):
        raise ValueError("NOISY and NORM must contain the same (level, seed) keys")
    for level, seed in keys:
        candidates = [str(value) for value in noisy[(level, seed)]]
        restored = [str(value) for value in normalized[(level, seed)]]
        rows.append(evaluate_transformation(
            references, candidates, condition=f"{level} noisy", noise_level=level,
            normalization=False, seed=seed, dataset="Hindi emotion dataset",
        ))
        rows.append(evaluate_transformation(
            references, restored, condition=f"{level} normalized", noise_level=level,
            normalization=True, seed=seed, dataset="Hindi emotion dataset",
        ))

    runs = pd.DataFrame(rows)
    summary = summarize_replicates(
        runs,
        ["condition", "noise_level", "normalization", "dataset", "language", "script"],
        ["bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge1", "rouge2", "rougeL", "meteor", "chrf", "ter"],
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        write_csv_new(runs, output_dir / "results_text_metrics_runs.csv"),
        write_csv_new(summary, output_dir / "results_text_metrics_summary.csv"),
    ]
    plots = plot_text_quality(summary, output_dir / "plots")
    return runs, summary, outputs, plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", type=Path, default=PROJECT_ROOT / "eval_sets.pkl",
        help="paired evaluation pickle produced by the project notebook",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "evaluation_outputs",
        help="new or existing directory for non-overwriting CSVs and plots",
    )
    args = parser.parse_args()
    _, summary, outputs, plots = evaluate_saved_pairs(args.pairs, args.output_dir)
    print(summary.to_string(index=False))
    print("CSV outputs:")
    print("\n".join(map(str, outputs)))
    print("Plots:")
    print("\n".join(map(str, plots)))


if __name__ == "__main__":
    main()
