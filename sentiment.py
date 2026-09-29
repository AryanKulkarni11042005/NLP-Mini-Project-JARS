import re, pickle, difflib, os
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "model"))
net = AutoModelForSequenceClassification.from_pretrained(os.path.join(HERE, "model")).to(DEV).eval()
N = pickle.load(open(os.path.join(HERE, "normalizer.pkl"), "rb"))
vocab_set, by_len, freq, classes = N["vocab_set"], N["by_len"], N["freq"], N["classes"]

@lru_cache(maxsize=None)
def _fix(w):
    if w in vocab_set:
        return w
    cands = [c for L in (len(w)-1, len(w), len(w)+1) for c in by_len.get(L, [])]
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

def predict(text):
    text = re.sub(r"[।?.,!\"'|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    if not devanagari:
        text = " ".join(_fix(w) for w in text.lower().split())
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64).to(DEV)
    with torch.no_grad():
        p = torch.softmax(net(**enc).logits, -1)[0]
    return {"label": classes[int(p.argmax())],
            "confidence": round(float(p.max()), 3),
            "probs": {c: round(float(p[i]), 3) for i, c in enumerate(classes)},
            "script": "Hindi" if devanagari else "Hinglish",
            "text_used": text}