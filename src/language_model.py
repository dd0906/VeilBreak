"""
A tiny character-level language model: "how English does this text look?"

Why we need it
--------------
Base64 can be checked structurally (it either decodes to readable text or it
doesn't). ROT13 and leetspeak cannot: ANY text can be ROT13'd, and ANY text
can have its '3's turned into 'e's. So we use a simple rule instead:

    "Apply the decoder. If the result looks MORE like English than the
     input did, the input was probably encoded."

"Looks like English" = average log-probability per character under a
trigram model (predict each character from the 2 before it), trained on
ordinary benign English prompts (Alpaca). Higher score = more English-like.

Trade-off (be ready to explain): this is a statistical heuristic, so it is
weaker on very short or non-English text. It is deterministic, needs no GPU,
and the whole model is a 31x31x31 table (~100 KB).

Text is first reduced to 31 symbols so the table stays small:
  a-z (26) | space | '#' for any digit | '.' for any punctuation | '?' for
  non-ASCII | '^' start-of-text padding.
Digits all map to '#' and symbols to '.', which is exactly what makes
leetspeak ("h4ck3r") score badly: letter->digit transitions are rare in English.
"""

import re

import numpy as np

# Symbol table: 26 letters, then the special symbols.
SYMBOLS = list("abcdefghijklmnopqrstuvwxyz") + [" ", "#", ".", "?", "^"]
INDEX = {s: i for i, s in enumerate(SYMBOLS)}
V = len(SYMBOLS)          # 31
BOS = INDEX["^"]

_WS = re.compile(r"\s+")


def normalize(text):
    """Map raw text onto the 31-symbol alphabet (see module docstring)."""
    out = []
    for ch in _WS.sub(" ", text.lower()):
        if "a" <= ch <= "z" or ch == " ":
            out.append(ch)
        elif ch.isdigit() and ch.isascii():
            out.append("#")
        elif ch.isascii():
            out.append(".")
        else:
            out.append("?")
    return "".join(out)


def _to_indices(text):
    """Normalized text -> array of symbol ids, with two start paddings."""
    ids = [BOS, BOS] + [INDEX[c] for c in normalize(text)]
    return np.array(ids, dtype=np.int64)


class CharLM:
    """Trigram character model with add-k smoothing."""

    def __init__(self, logp=None):
        self.logp = logp          # shape (V, V, V): log P(c | c-2, c-1)

    # ---- training ----
    def fit(self, texts, k=0.05):
        counts = np.zeros((V, V, V), dtype=np.float64)
        for t in texts:
            ids = _to_indices(t)
            if len(ids) < 3:
                continue
            np.add.at(counts, (ids[:-2], ids[1:-1], ids[2:]), 1)
        counts += k
        # Never predict the padding symbol as a next character.
        counts[:, :, BOS] = 1e-12
        self.logp = np.log(counts / counts.sum(axis=2, keepdims=True))
        return self

    # ---- scoring ----
    def score(self, text):
        """Average log-prob per character. Higher = more English-like."""
        ids = _to_indices(text)
        if len(ids) < 3:
            return float("-inf")
        lp = self.logp[ids[:-2], ids[1:-1], ids[2:]]
        return float(lp.mean())

    # ---- persistence ----
    def save(self, path):
        np.savez_compressed(path, logp=self.logp.astype(np.float32))

    @classmethod
    def load(cls, path):
        return cls(np.load(path)["logp"].astype(np.float64))
