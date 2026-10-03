import os
import re
import math
from collections import Counter

import pandas as pd

try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError:
    Language = None
    LanguageDetectorBuilder = None

import sentiment


# ---------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------

if LanguageDetectorBuilder is not None:
    # Lingua enum names can differ slightly across versions.
    # Add only languages that exist in the installed version.
    _language_names = [
        "ENGLISH",
        "HINDI",
        "MARATHI",
        "BENGALI",
        "GUJARATI",
        "PUNJABI",
        "TAMIL",
        "TELUGU",
        "KANNADA",
        "MALAYALAM",
        "URDU",
    ]

    LANGUAGES = [
        getattr(Language, name)
        for name in _language_names
        if hasattr(Language, name)
    ]

    DETECTOR = (
        LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
        if LANGUAGES
        else None
    )
else:
    DETECTOR = None

LANGUAGE_NAMES = {
    "ENGLISH": "English",
    "HINDI": "Hindi",
    "MARATHI": "Marathi",
    "BENGALI": "Bengali",
    "GUJARATI": "Gujarati",
    "PUNJABI": "Punjabi",
    "TAMIL": "Tamil",
    "TELUGU": "Telugu",
    "KANNADA": "Kannada",
    "MALAYALAM": "Malayalam",
    "URDU": "Urdu",
}

HINGLISH_HINTS = {
    "hai", "hain", "ka", "ki", "ke", "ko", "mein", "me", "ye", "yah",
    "mera", "meri", "mere", "bahut", "bohot", "acha", "achha", "accha",
    "kharab", "kharaab", "phone", "paisa", "vasool", "mujhe", "nahi",
    "nahin", "se", "par", "aur", "lekin", "wala", "wali", "sahi",
    "mast", "bekar", "bekaar", "pasand", "lagta", "lagti", "chahiye",
    "kimmat", "kimat", "quality", "camera", "battery", "display",
    "performance", "service", "product", "overall", "experience",
}


def _is_devanagari(text):
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def _roman_tokens(text):
    return re.findall(r"[A-Za-z]+", text.lower())


def analyze_code_mixing(text):
    tokens = _roman_tokens(text)
    if not tokens:
        return {"is_code_mixed": False, "code_mixed_score": 0.0}

    hint_hits = sum(t in HINGLISH_HINTS for t in tokens)
    devanagari = len(re.findall(r"[\u0900-\u097F]+", text))
    latin = len(re.findall(r"[A-Za-z]+", text))

    if devanagari and latin:
        score = 100.0
    elif hint_hits:
        score = min(100.0, round((hint_hits / max(1, len(tokens))) * 100, 2))
    else:
        score = 0.0

    mixed = bool(devanagari and latin) or (hint_hits >= 2 and len(tokens) >= 3)
    return {"is_code_mixed": mixed, "code_mixed_score": round(score, 2)}


def detect_language(text):
    text = str(text or "").strip()
    if not text:
        return "Unknown"

    mix = analyze_code_mixing(text)
    has_dev = _is_devanagari(text)
    latin_tokens = _roman_tokens(text)

    # Explicit Hinglish/code-mixing heuristic takes precedence.
    if mix["is_code_mixed"] and not has_dev:
        return "Hinglish"
    if mix["is_code_mixed"] and has_dev and latin_tokens:
        # Mixed Devanagari + Roman Hindi/English is treated as Hinglish/code-mixed.
        return "Hinglish"

    if DETECTOR is not None:
        try:
            detected = DETECTOR.detect_language_of(text)
            if detected is not None:
                return LANGUAGE_NAMES.get(detected.name, detected.name.title())
        except Exception:
            pass

    if has_dev:
        return "Hindi" if len(text) < 80 else "Hindi/Devanagari"

    return "English"


# ---------------------------------------------------------------------
# Tokenization / dataset NLP profile
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(r"[\u0900-\u097F]+|[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?")

def _tokens(text):
    return TOKEN_RE.findall(str(text).lower())


def _script(text):
    text = str(text)
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if devanagari and latin:
        return "Mixed"
    if devanagari:
        return "Devanagari"
    if latin:
        return "Latin"
    return "Other"


def analyze_dataset(df, text_column):
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found.")

    texts = df[text_column].fillna("").astype(str)
    token_lists = [_tokens(t) for t in texts]
    all_tokens = [tok for row in token_lists for tok in row]

    n = len(df)
    lengths = texts.str.len()

    if n:
        avg_chars = round(float(lengths.mean()), 2)
        min_chars = int(lengths.min())
        max_chars = int(lengths.max())
    else:
        avg_chars = min_chars = max_chars = 0

    token_lengths = [len(x) for x in token_lists]
    vocab = set(all_tokens)
    total_tokens = len(all_tokens)

    language_counts = Counter(detect_language(t) for t in texts)
    script_counts = Counter(_script(t) for t in texts)

    language_distribution = {
        k: round(v * 100 / n, 2) for k, v in language_counts.most_common()
    } if n else {}

    script_distribution = {
        k: round(v * 100 / n, 2) for k, v in script_counts.most_common()
    } if n else {}

    code_mix_scores = [analyze_code_mixing(t) for t in texts]
    mixed_count = sum(x["is_code_mixed"] for x in code_mix_scores)
    romanized_count = sum(
        bool(_roman_tokens(t)) and not _is_devanagari(t) for t in texts
    )

    duplicate_count = int(texts.duplicated().sum())
    punctuation_count = sum(len(re.findall(r"[^\w\s]", t, flags=re.UNICODE)) for t in texts)
    char_count = max(1, int(lengths.sum()))

    return {
        "documents": n,
        "avg_characters": avg_chars,
        "min_characters": min_chars,
        "max_characters": max_chars,
        "avg_tokens": round(sum(token_lengths) / max(1, n), 2),
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "vocabulary_size": len(vocab),
        "unique_token_ratio": round(len(vocab) * 100 / max(1, total_tokens), 2),
        "duplicate_documents": duplicate_count,
        "duplicate_percentage": round(duplicate_count * 100 / max(1, n), 2),
        "emoji_count": sum(
            len(re.findall(r"[\U0001F300-\U0001FAFF]", t)) for t in texts
        ),
        "punctuation_density": round(punctuation_count * 100 / char_count, 2),
        "primary_language": max(language_distribution, key=language_distribution.get)
        if language_distribution else "Unknown",
        "language_distribution": language_distribution,
        "script_distribution": script_distribution,
        "code_mixed_percentage": round(mixed_count * 100 / max(1, n), 2),
        "romanized_percentage": round(romanized_count * 100 / max(1, n), 2),
        "top_tokens": Counter(all_tokens).most_common(10),
    }


# ---------------------------------------------------------------------
# Column / task detection
# ---------------------------------------------------------------------

TEXT_HINTS = ("text", "review", "comment", "sentence", "content", "message", "tweet", "description")
LABEL_HINTS = ("label", "sentiment", "target", "class", "category", "polarity")

SENTIMENT_LABELS = {
    "positive", "negative", "neutral",
    "pos", "neg", "neu",
    "1", "0", "2",
}


def detect_text_column(df):
    candidates = []
    for col in df.columns:
        name = str(col).lower().strip()
        if any(h in name for h in TEXT_HINTS):
            candidates.append(col)
    if candidates:
        return candidates[0]

    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if object_cols:
        return max(object_cols, key=lambda c: df[c].fillna("").astype(str).str.len().mean())
    return df.columns[0] if len(df.columns) else None


def detect_label_column(df, text_column=None):
    for col in df.columns:
        if col == text_column:
            continue
        name = str(col).lower().strip()
        if any(h in name for h in LABEL_HINTS):
            return col

    for col in df.columns:
        if col == text_column:
            continue
        values = set(df[col].dropna().astype(str).str.lower().str.strip().unique())
        if values and values.issubset(SENTIMENT_LABELS):
            return col
    return None


def detect_task(df, label_column=None):
    if not label_column:
        return "Unknown / unlabeled"

    values = set(df[label_column].dropna().astype(str).str.lower().str.strip().unique())
    if values and values.issubset(SENTIMENT_LABELS):
        return "Sentiment classification"

    return "Classification (non-sentiment labels)"


# ---------------------------------------------------------------------
# Existing-model benchmark
# ---------------------------------------------------------------------

def _normalize_label(x):
    x = str(x).strip().lower()
    aliases = {
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
    }
    return aliases.get(x, x)


def benchmark_models(df, text_column, label_column):
    """Evaluate only the already-trained sentiment models. No retraining."""
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found.")

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    texts = df[text_column].fillna("").astype(str).tolist()
    y_true = [_normalize_label(x) for x in df[label_column].tolist()]

    models = sentiment.available_models()
    if not models:
        raise RuntimeError("No trained models are available.")

    # Warm up each model once so first-load time is not mixed into inference timing.
    for model in models:
        sentiment.predict(texts[0] if texts else "", model)

    rows = []
    predictions = {}

    for model in models:
        preds = []
        times = []

        for text in texts:
            result = sentiment.predict(text, model)
            preds.append(_normalize_label(result["label"]))
            times.append(float(result["ms"]))

        predictions[model] = preds

        rows.append({
            "Model": model,
            "Accuracy": round(accuracy_score(y_true, preds), 4),
            "Precision": round(precision_score(y_true, preds, average="macro", zero_division=0), 4),
            "Recall": round(recall_score(y_true, preds, average="macro", zero_division=0), 4),
            "Macro F1": round(f1_score(y_true, preds, average="macro", zero_division=0), 4),
            "Weighted F1": round(f1_score(y_true, preds, average="weighted", zero_division=0), 4),
            "Avg Time (ms)": round(sum(times) / max(1, len(times)), 3),
        })

    result_df = pd.DataFrame(rows)

    confusion = {}
    labels = sorted(set(y_true))
    for model, preds in predictions.items():
        cm = confusion_matrix(y_true, preds, labels=labels)
        confusion[model] = {
            "labels": labels,
            "matrix": cm.tolist(),
        }

    return result_df, confusion


# ---------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------

def _minmax_inverse(series):
    s = pd.Series(series, dtype=float)
    if len(s) == 1 or s.max() == s.min():
        return pd.Series([1.0] * len(s), index=s.index)
    return (s.max() - s) / (s.max() - s.min())


def rank_models(benchmark_df, priority="Balanced"):
    df = benchmark_df.copy()

    # Performance score emphasizes Macro-F1 to avoid rewarding only majority-class behavior.
    df["Quality Score"] = (
        0.55 * df["Macro F1"] +
        0.25 * df["Weighted F1"] +
        0.20 * df["Accuracy"]
    )

    df["Speed Score"] = _minmax_inverse(df["Avg Time (ms)"])

    if priority == "Accuracy":
        df["Recommendation Score"] = df["Quality Score"]
    elif priority == "Speed":
        df["Recommendation Score"] = (
            0.75 * df["Speed Score"] +
            0.25 * df["Quality Score"]
        )
    else:
        df["Recommendation Score"] = (
            0.70 * df["Quality Score"] +
            0.30 * df["Speed Score"]
        )

    df = df.sort_values(
        ["Recommendation Score", "Macro F1"],
        ascending=False
    ).reset_index(drop=True)

    df.insert(0, "Rank", range(1, len(df) + 1))
    df["Recommendation Score"] = df["Recommendation Score"].round(4)
    df["Quality Score"] = df["Quality Score"].round(4)
    df["Speed Score"] = df["Speed Score"].round(4)
    return df


def recommend_from_dataset(
    df,
    text_column=None,
    label_column=None,
    priority="Balanced",
):
    text_column = text_column or detect_text_column(df)
    if not text_column:
        raise ValueError("Could not identify a text column.")

    label_column = label_column or detect_label_column(df, text_column)
    profile = analyze_dataset(df, text_column)
    task = detect_task(df, label_column)

    result = {
        "text_column": text_column,
        "label_column": label_column,
        "task": task,
        "profile": profile,
        "benchmark": None,
        "ranking": None,
        "recommendation": None,
        "reason": None,
        "confusion": None,
    }

    # Existing models are sentiment classifiers only.
    if task == "Sentiment classification":
        benchmark_df, confusion = benchmark_models(df, text_column, label_column)
        ranking = rank_models(benchmark_df, priority)
        recommended = ranking.iloc[0]["Model"]

        result["benchmark"] = benchmark_df
        result["ranking"] = ranking
        result["confusion"] = confusion
        result["recommendation"] = recommended

        top = ranking.iloc[0]
        if priority == "Accuracy":
            reason = (
                f"{recommended} has the highest recommendation score for Accuracy priority, "
                f"with Macro-F1={top['Macro F1']:.4f} and Accuracy={top['Accuracy']:.4f}."
            )
        elif priority == "Speed":
            reason = (
                f"{recommended} has the highest recommendation score for Speed priority, "
                f"balancing inference speed ({top['Avg Time (ms)']:.3f} ms) with measured quality."
            )
        else:
            reason = (
                f"{recommended} has the highest Balanced score, combining measured quality "
                f"(Macro-F1={top['Macro F1']:.4f}) and inference speed ({top['Avg Time (ms)']:.3f} ms)."
            )
        result["reason"] = reason
    else:
        # No fabricated accuracy: there is no compatible trained model for an unknown task.
        result["recommendation"] = "No compatible existing model"
        result["reason"] = (
            "The current project contains sentiment-classification models only. "
            "Because this dataset is not identified as sentiment-labeled, "
            "accuracy/F1-based model recommendation is not claimed."
        )

    return result
