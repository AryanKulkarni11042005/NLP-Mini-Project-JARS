"""The export notebook's deterministic text preparation and corruption rules."""

from __future__ import annotations

import difflib
import random
import re
import unicodedata
from collections import Counter
from functools import lru_cache


SWAPS = [
    ("aa", ["a"]), ("a", ["aa"]), ("ee", ["i"]), ("i", ["ee", "y"]),
    ("oo", ["u"]), ("u", ["oo"]), ("kh", ["k"]), ("ch", ["chh"]),
    ("th", ["t"]), ("dh", ["d"]), ("v", ["w"]), ("sh", ["s"]),
]
NOISE_LEVELS = {"Low": 0.3, "Medium": 0.6, "High": 1.0}
NOISE_SEEDS = [1, 2, 3, 4, 5]


def clean_hindi(text: str) -> str:
    text = re.sub(r"[।?.,!\"'|]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def to_hinglish(text: str) -> str:
    """Replicate the export notebook's Devanagari-to-ITRANS mapping."""
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    t = unicodedata.normalize("NFD", str(text)).replace("\u093c", "")
    t = unicodedata.normalize("NFC", t)
    t = t.replace("ऑ", "ओ").replace("ॉ", "ो").replace("ॅ", "े")
    s = transliterate(t, sanscript.DEVANAGARI, sanscript.ITRANS)
    words = []
    for word in s.split():
        if len(word) > 2 and word.endswith("a") and word[-2] not in "aeiouAEIOU":
            word = word[:-1]
        words.append(word)
    s = " ".join(words)
    s = s.replace(".N", "n").replace("RRi", "ri")
    s = s.replace("M", "n").replace("~N", "n").replace("N", "n")
    s = s.replace("aa", "a").replace("A", "a").replace("I", "i").replace("U", "u")
    s = s.lower()
    s = re.sub(r"[^\x00-\x7f]", "", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return re.sub(r"\s+", " ", s).strip()


def add_noise(text: str, level: float, seed: int) -> str:
    """Use the export notebook's stable per-text/per-seed noise procedure."""
    if not 0.0 <= level <= 1.0:
        raise ValueError("Noise level must be between 0 and 1")
    rng = random.Random(seed * 1000003 + sum(map(ord, text)))
    out = []
    for word in text.split():
        if rng.random() < level:
            pat, reps = rng.choice(SWAPS)
            if pat in word:
                word = word.replace(pat, rng.choice(reps), 1)
            elif len(word) > 3:
                i = rng.randrange(1, len(word))
                word = word[:i] + word[i + 1:]
            elif len(word) > 1:
                word = word + word[-1]
        out.append(word)
    return " ".join(out)


def fit_normalizer(training_texts, vocab_limit: int = 30000):
    """Fit the notebook's train-only edit-distance normalizer.

    Returns (normalize_callable, artifact_state). The artifact state has the
    same keys consumed by sentiment.py, allowing the existing model export to
    keep its format.
    """
    freq = Counter(w for sentence in training_texts for w in str(sentence).split())
    vocab = [w for w, _ in freq.most_common(vocab_limit)]
    vocab_set = set(vocab)
    by_len = {}
    for word in vocab:
        by_len.setdefault(len(word), []).append(word)

    @lru_cache(maxsize=None)
    def normalize_word(word):
        if word in vocab_set:
            return word
        cands = [c for length in (len(word) - 1, len(word), len(word) + 1) for c in by_len.get(length, [])]
        close = difflib.get_close_matches(word, cands, n=5, cutoff=0.75)
        if not close:
            return word
        matcher = difflib.SequenceMatcher(None, word)
        scored = []
        for candidate in close:
            matcher.set_seq2(candidate)
            scored.append((matcher.ratio(), candidate))
        best = max(score for score, _ in scored)
        top = [candidate for score, candidate in scored if score >= best - 0.08]
        return max(top, key=lambda candidate: freq[candidate])

    def normalize(text):
        return " ".join(normalize_word(word) for word in str(text).split())

    artifact = {"freq": freq, "vocab_set": vocab_set, "by_len": by_len, "vocab": vocab}
    return normalize, artifact
