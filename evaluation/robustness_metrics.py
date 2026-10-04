"""Seed aggregation, robustness deltas, and exploratory quality correlations."""

from __future__ import annotations

import pandas as pd
from scipy.stats import pearsonr, spearmanr


CLASSIFICATION_METRICS = (
    "accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"
)


def summarize_replicates(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]):
    """Mean and sample standard deviation; singleton SD is left as NaN."""
    if frame.empty:
        return frame.copy()
    result = frame.groupby(group_columns, dropna=False)[metrics].agg(["mean", "std"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    return result.reset_index()


def robustness_table(classification_runs: pd.DataFrame) -> pd.DataFrame:
    """Build per-model Macro-F1 conditions and high-noise drop/recovery."""
    if classification_runs.empty:
        return pd.DataFrame()
    means = classification_runs.groupby(["model", "condition"], as_index=False)["macro_f1"].mean()
    wide = means.pivot(index="model", columns="condition", values="macro_f1")
    required = ["Clean Hinglish", "Low noisy", "Medium noisy", "High noisy", "High normalized"]
    for name in required:
        if name not in wide:
            wide[name] = float("nan")
    clean, high_noise, high_norm = (
        wide["Clean Hinglish"], wide["High noisy"], wide["High normalized"]
    )
    absolute_drop = clean - high_noise
    wide["F1 drop (clean - high noise)"] = absolute_drop
    wide["F1 drop (%)"] = 100.0 * absolute_drop / clean.where(clean != 0)
    wide["F1 recovered"] = high_norm - high_noise
    wide["F1 recovery (%)"] = 100.0 * (high_norm - high_noise) / absolute_drop.where(absolute_drop != 0)
    return wide.reset_index()


def exploratory_correlations(
    text_runs: pd.DataFrame,
    classification_runs: pd.DataFrame,
    quality_metrics=("chrf", "bleu", "rougeL"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair text quality and Macro-F1 by condition/seed, without p-value claims.

    Returns (correlation table, paired scatter data). Because all conditions
    share one held-out corpus and noise levels are repeated, results are
    explicitly exploratory rather than inferential.
    """
    if text_runs.empty or classification_runs.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["condition", "noise_level", "normalization", "seed"]
    available = [m for m in quality_metrics if m in text_runs.columns]
    text = text_runs[keys + available].copy()
    cls = classification_runs[
        ["model", *keys, "macro_f1"]
    ].copy()
    # Correlate the transformation conditions of interest (noise and
    # restoration); the clean identity baseline is not a transformed output.
    text = text[text["noise_level"].astype(str).str.lower() != "none"]
    cls = cls[cls["noise_level"].astype(str).str.lower() != "none"]
    paired = cls.merge(text, on=keys, how="inner", validate="many_to_one")
    rows = []
    for model, model_frame in paired.groupby("model"):
        for metric in available:
            points = model_frame[[metric, "macro_f1"]].dropna()
            if len(points) < 3:
                continue
            if points[metric].nunique() < 2 or points["macro_f1"].nunique() < 2:
                pearson = spearman = float("nan")
            else:
                pearson = float(pearsonr(points[metric], points["macro_f1"]).statistic)
                spearman = float(spearmanr(points[metric], points["macro_f1"]).statistic)
            rows.append({
                "model": model,
                "text_metric": metric,
                "n_paired_conditions": len(points),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "interpretation": "exploratory; repeated conditions share the same held-out corpus",
            })
    return pd.DataFrame(rows), paired
