"""Language-agnostic text-transformation metrics.

BLEU, chrF, TER use SacreBLEU. ROUGE uses rouge-score without stemming.
METEOR uses NLTK's alignment/fragmentation formula with exact token matches
only (no English stemmer or WordNet). Text is never transliterated or
normalized here; tokenization is explicit and Unicode-aware.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable, Mapping

import sacrebleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer


METRIC_COLUMNS = (
    "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "rouge1", "rouge2", "rougeL", "meteor", "chrf", "ter",
)
TOKENIZER_NAME = "Unicode category tokenizer: letters/numbers/marks plus punctuation"


def tokenize_text(text: str) -> list[str]:
    """Split Unicode letters/numbers/marks and preserve punctuation tokens.

    Python's built-in ``re`` treats combining marks inconsistently as ``\w``;
    grouping Unicode mark categories with their base letters keeps Indic
    vowel signs and virama attached to the word.
    """
    tokens = []
    current = []
    for char in str(text):
        category = unicodedata.category(char)
        if char == "_" or category[0] in ("L", "N") or (category[0] == "M" and current):
            current.append(char)
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


class _UnicodeTokenizer:
    def tokenize(self, text):
        return tokenize_text(text)


class _IdentityStemmer:
    """Disable language-specific stemming while retaining METEOR's API."""
    @staticmethod
    def stem(word):
        return word


class _NoSynonyms:
    """WordNet-compatible empty lookup to avoid English-only matches."""
    @staticmethod
    def synsets(_word):
        return []


def _aligned_texts(references: Iterable[str], candidates: Iterable[str]):
    refs = ["" if x is None else str(x) for x in references]
    hyps = ["" if x is None else str(x) for x in candidates]
    if len(refs) != len(hyps):
        raise ValueError(
            f"Reference/candidate length mismatch: {len(refs)} != {len(hyps)}"
        )
    if not refs:
        raise ValueError("At least one aligned reference/candidate pair is required")
    return refs, hyps


def _corpus_bleu(references: list[str], candidates: list[str], max_order: int = 4):
    pairs = [(r, h) for r, h in zip(references, candidates) if r or h]
    if not pairs:
        return 100.0
    refs, hyps = zip(*pairs)
    metric = sacrebleu.metrics.BLEU(
        lowercase=False,
        tokenize="none",  # Inputs are whitespace-joined Unicode regex tokens.
        smooth_method="exp",
        max_ngram_order=max_order,
        effective_order=True,
    )
    score = float(metric.corpus_score(list(hyps), [list(refs)]).score)
    return round(min(100.0, max(0.0, score)), 8)


def calculate_bleu(references: Iterable[str], candidates: Iterable[str]) -> float:
    refs, hyps = _aligned_texts(references, candidates)
    tok_refs = [" ".join(tokenize_text(x)) for x in refs]
    tok_hyps = [" ".join(tokenize_text(x)) for x in hyps]
    return _corpus_bleu(tok_refs, tok_hyps, 4)


def calculate_bleu_n(
    references: Iterable[str], candidates: Iterable[str], n: int
) -> float:
    if n not in (1, 2, 3, 4):
        raise ValueError("BLEU n-gram order must be between 1 and 4")
    refs, hyps = _aligned_texts(references, candidates)
    tok_refs = [" ".join(tokenize_text(x)) for x in refs]
    tok_hyps = [" ".join(tokenize_text(x)) for x in hyps]
    return _corpus_bleu(tok_refs, tok_hyps, n)


def calculate_rouge(references: Iterable[str], candidates: Iterable[str]):
    refs, hyps = _aligned_texts(references, candidates)
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False,
        tokenizer=_UnicodeTokenizer(),
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, hyp in zip(refs, hyps):
        ref_tokens, hyp_tokens = tokenize_text(ref), tokenize_text(hyp)
        if not ref_tokens and not hyp_tokens:
            scores = {k: 1.0 for k in totals}
        elif not ref_tokens or not hyp_tokens:
            scores = {k: 0.0 for k in totals}
        elif ref_tokens == hyp_tokens:
            scores = {k: 1.0 for k in totals}
        else:
            scores = scorer.score(ref, hyp)
            scores = {k: scores[k].fmeasure for k in totals}
        for key in totals:
            totals[key] += scores[key]
    scale = 100.0 / len(refs)
    return {key: round(value * scale, 8) for key, value in totals.items()}


def calculate_meteor(references: Iterable[str], candidates: Iterable[str]) -> float:
    """Mean sentence METEOR (0..100), using exact tokens only.

    This avoids applying English Porter stemming or requiring English
    WordNet for Hindi/Hinglish. METEOR remains a lexical, not semantic, score.
    """
    refs, hyps = _aligned_texts(references, candidates)
    scores = []
    for ref, hyp in zip(refs, hyps):
        ref_tokens, hyp_tokens = tokenize_text(ref), tokenize_text(hyp)
        if not ref_tokens and not hyp_tokens:
            scores.append(1.0)
        elif not ref_tokens or not hyp_tokens:
            scores.append(0.0)
        elif ref_tokens == hyp_tokens:
            scores.append(1.0)
        else:
            scores.append(single_meteor_score(
                ref_tokens, hyp_tokens,
                stemmer=_IdentityStemmer(), wordnet=_NoSynonyms()
            ))
    return round(100.0 * sum(scores) / len(scores), 8)


def calculate_chrf(references: Iterable[str], candidates: Iterable[str]) -> float:
    refs, hyps = _aligned_texts(references, candidates)
    pairs = [(r, h) for r, h in zip(refs, hyps) if r or h]
    if not pairs:
        return 100.0
    refs, hyps = zip(*pairs)
    metric = sacrebleu.metrics.CHRF(
        char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False
    )
    score = float(metric.corpus_score(list(hyps), [list(refs)]).score)
    return round(min(100.0, max(0.0, score)), 8)


def calculate_ter(references: Iterable[str], candidates: Iterable[str]) -> float:
    """SacreBLEU TER (lower is better), using whitespace token boundaries."""
    refs, hyps = _aligned_texts(references, candidates)
    pairs = [(r, h) for r, h in zip(refs, hyps) if r or h]
    if not pairs:
        return 0.0
    refs, hyps = zip(*pairs)
    metric = sacrebleu.metrics.TER(
        normalized=False, no_punct=False, asian_support=False, case_sensitive=True
    )
    return float(metric.corpus_score(list(hyps), [list(refs)]).score)


def calculate_text_metrics(
    references: Iterable[str], candidates: Iterable[str]
) -> dict[str, float]:
    """Calculate the supported corpus and mean sentence metrics (all x100)."""
    refs, hyps = _aligned_texts(references, candidates)
    rouge = calculate_rouge(refs, hyps)
    result = {
        "bleu": calculate_bleu(refs, hyps),
        **{f"bleu_{n}": calculate_bleu_n(refs, hyps, n) for n in range(1, 5)},
        **rouge,
        "meteor": calculate_meteor(refs, hyps),
        "chrf": calculate_chrf(refs, hyps),
        "ter": calculate_ter(refs, hyps),
    }
    return result


def evaluate_transformation(
    references: Iterable[str],
    candidates: Iterable[str],
    *,
    condition: str,
    noise_level: str,
    normalization: bool,
    seed: int | str,
    dataset: str,
    language: str = "Hinglish",
    script: str = "Latin",
) -> dict:
    """Return one reproducible, aligned experiment replicate."""
    refs, hyps = _aligned_texts(references, candidates)
    row = {
        "task": "paired text restoration",
        "condition": condition,
        "candidate": "noisy or normalized Hinglish",
        "reference": "paired clean Hinglish generated from held-out Hindi",
        "dataset": dataset,
        "language": language,
        "script": script,
        "noise_level": noise_level,
        "normalization": bool(normalization),
        "seed": seed,
        "n_samples": len(refs),
        "tokenizer": TOKENIZER_NAME,
    }
    row.update(calculate_text_metrics(refs, hyps))
    return row
