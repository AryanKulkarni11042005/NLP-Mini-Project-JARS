import os, re, pickle, difflib, time, glob
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_NAMES = ["IndicBERT", "TF-IDF + LogReg", "LSTM", "fastText"]

def _model_artifact(model, subdir, filename):
    """Find a model in the legacy layout or a timestamped export bundle."""
    expected = os.path.join(MODEL_DIR, subdir, filename)
    if os.path.isfile(expected):
        return expected

    # Some training/export runs keep each model under a timestamped folder,
    # e.g. model/tfidf-<run-id>/tfidf/tfidf.joblib.
    candidates = glob.glob(os.path.join(
        MODEL_DIR, f"{subdir}-*", subdir, filename
    ))
    if not candidates:
        return expected
    return max(candidates, key=os.path.getmtime)


PATHS = {
    "IndicBERT": _model_artifact("IndicBERT", "indicBERT", "config.json"),
    "TF-IDF + LogReg": _model_artifact("tfidf", "tfidf", "tfidf.joblib"),
    "LSTM": _model_artifact("lstm", "lstm", "lstm.pt"),
    "fastText": _model_artifact("fasttext", "fasttext", "fasttext.bin"),
}


def _normalizer_path():
    for p in (os.path.join(MODEL_DIR, "normalizer.pkl"), os.path.join(HERE, "normalizer.pkl")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("normalizer.pkl not found in model/ or the project root")


_N = pickle.load(open(_normalizer_path(), "rb"))
vocab_set, by_len, freq, classes = _N["vocab_set"], _N["by_len"], _N["freq"], _N["classes"]


@lru_cache(maxsize=None)
def _fix(w):
    if w in vocab_set:
        return w
    cands = [c for L in (len(w) - 1, len(w), len(w) + 1) for c in by_len.get(L, [])]
    close = difflib.get_close_matches(w, cands, n=5, cutoff=0.75)
    if not close:
        return w
    sm = difflib.SequenceMatcher(None, w)
    scored = []
    for c in close:
        sm.set_seq2(c)
        scored.append((sm.ratio(), c))
    best = max(s for s, _ in scored)
    return max([c for s, c in scored if s >= best - 0.08], key=lambda c: freq[c])


def prepare(text):
    text = re.sub(r"[।?.,!\"'|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    devanagari = bool(re.search(r"[ऀ-ॿ]", text))
    if not devanagari:
        text = " ".join(_fix(w) for w in text.lower().split())
    return text, devanagari


_cache = {}
_errors = {}


def _load_indicbert():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.dirname(PATHS["IndicBERT"])
    tok = AutoTokenizer.from_pretrained(checkpoint_dir)
    net = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(dev).eval()

    def run(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64).to(dev)
        with torch.no_grad():
            p = torch.softmax(net(**enc).logits, -1)[0].cpu().numpy()
        return {c: float(p[i]) for i, c in enumerate(classes)}

    return run


def _load_tfidf():
    import joblib

    d = joblib.load(PATHS["TF-IDF + LogReg"])
    vec, model = d["vec"], d["model"]

    def run(text):
        p = model.predict_proba(vec.transform([text]))[0]
        return {c: float(p[i]) for i, c in enumerate(model.classes_)}

    return run


def _load_lstm():
    import torch
    import torch.nn as nn

    class LSTMNet(nn.Module):
        def __init__(self, vocab_size, n_classes):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 100, padding_idx=0)
            self.lstm = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.4)
            self.fc = nn.Linear(256, n_classes)

        def forward(self, x):
            mask = (x != 0).unsqueeze(-1)
            h, _ = self.lstm(self.emb(x))
            return self.fc(self.drop(h.masked_fill(~mask, -1e9).max(1).values))

    ck = torch.load(PATHS["LSTM"], map_location="cpu")
    itos, lcls, maxlen = ck["itos"], ck["classes"], ck["maxlen"]
    stoi = {w: i for i, w in enumerate(itos)}
    net = LSTMNet(len(itos), len(lcls))
    net.load_state_dict(ck["state_dict"])
    net.eval()

    def run(text):
        ids = [stoi.get(w, 1) for w in text.split()][:maxlen]
        x = torch.zeros(1, maxlen, dtype=torch.long)
        if ids:
            x[0, :len(ids)] = torch.tensor(ids)
        with torch.no_grad():
            p = torch.softmax(net(x), -1)[0].numpy()
        return {c: float(p[i]) for i, c in enumerate(lcls)}

    return run


def _load_fasttext():
    import fasttext

    ft = fasttext.load_model(PATHS["fastText"])
    k = len(ft.get_labels())

    def run(text):
        out = ft.f.predict(text.replace("\n", " ") + "\n", k, 0.0, "strict")
        probs = {lab.replace("__label__", ""): float(pr) for pr, lab in out}
        for c in classes:
            probs.setdefault(c, 0.0)
        s = sum(probs.values()) or 1.0
        return {c: probs[c] / s for c in classes}

    return run


_LOADERS = {
    "IndicBERT": _load_indicbert,
    "TF-IDF + LogReg": _load_tfidf,
    "LSTM": _load_lstm,
    "fastText": _load_fasttext,
}


def available_models():
    return [m for m in MODEL_NAMES if os.path.exists(PATHS[m])]


def _get(model):
    if model in _cache:
        return _cache[model]
    if model not in _LOADERS:
        raise ValueError(f"unknown model: {model}")
    if not os.path.exists(PATHS[model]):
        raise FileNotFoundError(f"{model}: {PATHS[model]} not found")
    fn = _LOADERS[model]()
    _cache[model] = fn
    return fn


def predict(text, model="IndicBERT"):
    used, devanagari = prepare(text)
    t0 = time.perf_counter()
    probs = _get(model)(used)
    ms = (time.perf_counter() - t0) * 1000
    label = max(probs, key=probs.get)
    return {
        "model": model,
        "label": label,
        "confidence": round(probs[label], 3),
        "probs": {c: round(probs.get(c, 0.0), 3) for c in classes},
        "script": "Hindi" if devanagari else "Hinglish",
        "text_used": used,
        "ms": round(ms, 1),
    }


def compare(text):
    rows = []
    for m in available_models():
        try:
            r = predict(text, m)
            rows.append(r)
        except Exception as e:
            _errors[m] = str(e)
            rows.append({"model": m, "label": "error", "confidence": 0.0,
                         "probs": {c: 0.0 for c in classes}, "script": "", "text_used": "",
                         "ms": 0.0, "error": str(e)})
    return rows
