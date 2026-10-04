"""Classification metrics kept separate from text-transformation scores."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_scores(y_true, y_pred) -> dict[str, float]:
    truth, predictions = list(y_true), list(y_pred)
    if len(truth) != len(predictions):
        raise ValueError(
            f"Prediction/reference length mismatch: {len(predictions)} != {len(truth)}"
        )
    if not truth:
        raise ValueError("At least one labeled example is required")
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision_macro": float(precision_score(
            truth, predictions, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            truth, predictions, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            truth, predictions, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            truth, predictions, average="weighted", zero_division=0
        )),
    }


def evaluate_classification_conditions(
    y_true,
    predictors: dict,
    conditions: dict,
    *,
    dataset: str,
    language: str,
    script: str,
) -> list[dict]:
    """Evaluate model callables across aligned named text conditions.

    `conditions` maps condition names to metadata and a candidate text sequence.
    Use an iterable of one-seed records for noisy conditions so repeated-noise
    uncertainty remains available in the returned rows.
    """
    rows = []
    for condition_name, replicates in conditions.items():
        for replicate in replicates:
            texts = list(replicate["texts"])
            if len(texts) != len(y_true):
                raise ValueError(
                    f"{condition_name}: {len(texts)} texts for {len(y_true)} labels"
                )
            for model_name, predict in predictors.items():
                predictions = list(predict(texts))
                metrics = classification_scores(y_true, predictions)
                rows.append({
                    "model": model_name,
                    "dataset": dataset,
                    "language": replicate.get("language", language),
                    "script": replicate.get("script", script),
                    "condition": condition_name,
                    "noise_level": replicate.get("noise_level", "none"),
                    "normalization": bool(replicate.get("normalization", False)),
                    "seed": replicate.get("seed", "clean"),
                    "n_samples": len(texts),
                    **metrics,
                })
    return rows
