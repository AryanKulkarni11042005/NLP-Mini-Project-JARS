"""Presentation-friendly plots for classification robustness and text quality."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _save_new(fig, output_dir: str | Path, name: str) -> Path:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if target.exists():
        target = folder / f"{target.stem}_2{target.suffix}"
        suffix = 2
        while target.exists():
            suffix += 1
            target = folder / f"{Path(name).stem}_{suffix}{Path(name).suffix}"
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_classification_robustness(summary: pd.DataFrame, output_dir):
    conditions = [
        "Clean Hinglish", "Low noisy", "Medium noisy", "High noisy",
        "Low normalized", "Medium normalized", "High normalized",
    ]
    data = summary[summary["condition"].isin(conditions)]
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, rows in data.groupby("model"):
        rows = rows.set_index("condition").reindex(conditions)
        ax.plot(conditions, rows["macro_f1_mean"], marker="o", linewidth=2, label=model)
        sd = rows["macro_f1_std"]
        if sd.notna().any():
            ax.fill_between(
                range(len(conditions)),
                (rows["macro_f1_mean"] - sd).to_numpy(dtype=float),
                (rows["macro_f1_mean"] + sd).to_numpy(dtype=float),
                alpha=0.10,
            )
    ax.set(title="Sentiment classification robustness", ylabel="Macro F1", xlabel="Input condition")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model")
    return _save_new(fig, output_dir, "classification_macro_f1.png")


def plot_normalization_recovery(classification_runs: pd.DataFrame, output_dir):
    data = classification_runs[
        classification_runs["condition"].isin(
            ["Low noisy", "Medium noisy", "High noisy", "Low normalized", "Medium normalized", "High normalized"]
        )
    ]
    if data.empty:
        return None
    keys = ["model", "noise_level", "seed"]
    noisy = data[~data["normalization"]][keys + ["macro_f1"]].rename(columns={"macro_f1": "f1_noisy"})
    norm = data[data["normalization"]][keys + ["macro_f1"]].rename(columns={"macro_f1": "f1_normalized"})
    paired = noisy.merge(norm, on=keys, validate="one_to_one")
    paired["recovery"] = paired["f1_normalized"] - paired["f1_noisy"]
    summary = paired.groupby(["model", "noise_level"])["recovery"].agg(["mean", "std"]).reset_index()
    order = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for model, rows in summary.groupby("model"):
        rows = rows.set_index("noise_level").reindex(order)
        ax.errorbar(order, rows["mean"], yerr=rows["std"], marker="o", linewidth=2, capsize=3, label=model)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Macro-F1 change after normalization", ylabel="Normalized F1 − noisy F1", xlabel="Noise level")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model")
    return _save_new(fig, output_dir, "normalization_recovery.png")


def plot_text_quality(text_summary: pd.DataFrame, output_dir):
    if text_summary.empty:
        return []
    ordered = [
        "Clean Hinglish", "Low noisy", "Medium noisy", "High noisy",
        "Low normalized", "Medium normalized", "High normalized",
    ]
    metrics = ["bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge1", "rouge2", "rougeL", "meteor", "chrf"]
    data = text_summary[text_summary["condition"].isin(ordered)]
    present = [m for m in metrics if f"{m}_mean" in data]
    if not present or data.empty:
        return []
    display_names = {
        "bleu": "BLEU", "bleu_1": "BLEU-1", "bleu_2": "BLEU-2",
        "bleu_3": "BLEU-3", "bleu_4": "BLEU-4",
        "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
        "meteor": "METEOR", "chrf": "chrF",
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    for metric in present:
        rows = data.set_index("condition").reindex(ordered)
        ax.plot(ordered, rows[f"{metric}_mean"], marker="o", label=display_names.get(metric, metric))
    ax.set(title="Text transformation metrics (higher is better)", ylabel="Score (0–100)", xlabel="Candidate condition")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    paths = [_save_new(fig, output_dir, "text_metrics_by_condition.png")]

    if "ter_mean" in data:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        rows = data.set_index("condition").reindex(ordered)
        ax.plot(ordered, rows["ter_mean"], marker="o", color="#9c3d10", linewidth=2)
        ax.set(title="Translation Edit Rate (lower is better)", ylabel="TER (0–100)", xlabel="Candidate condition")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        paths.append(_save_new(fig, output_dir, "ter_by_condition.png"))
    return paths


def plot_quality_correlations(paired: pd.DataFrame, output_dir, minimum_points=6):
    if paired.empty:
        return []
    metrics = [m for m in ("chrf", "bleu", "rougeL") if m in paired]
    paths = []
    for metric in metrics:
        points = paired[[metric, "macro_f1"]].dropna()
        if len(points) < minimum_points or points[metric].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for model, rows in paired.groupby("model"):
            rows = rows[[metric, "macro_f1"]].dropna()
            if not rows.empty:
                ax.scatter(rows[metric], rows["macro_f1"], alpha=0.6, label=model)
        ax.set(title=f"Exploratory: {metric.upper()} vs sentiment Macro F1", xlabel=f"{metric.upper()} (0–100)", ylabel="Macro F1")
        ax.grid(alpha=0.25)
        ax.legend(title="Model")
        paths.append(_save_new(fig, output_dir, f"quality_vs_macro_f1_{metric}.png"))
    return paths
