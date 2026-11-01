"""
Safety classifier + the full VeilBreak guard (decode, then classify).

SafetyClassifier
    text --> embedding vector --> logistic regression --> safe / unsafe

    "Embedding" = turn text into a list of numbers so that similar meanings
    end up with similar numbers. Logistic regression then draws one line
    through that space: unsafe on one side, safe on the other.

    Two embedders are available:
      'minilm' : sentence-transformers all-MiniLM-L6-v2 (384 numbers). The
                 real one: understands meaning, runs on a CPU.
      'tfidf'  : character n-gram counts. No download needed -- used for
                 tests and quick smoke runs, and as a weaker reference point.

    IMPORTANT (scope rule): the classifier is trained on PLAIN text only.
    It never sees an encoded prompt during training. Handling encodings is
    the decoder's job, not the classifier's.

VeilBreakGuard
    prompt --> decode() --> SafetyClassifier --> Verdict
"""

import os
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.pipeline import DecodeResult, decode

LABELS = ["safe", "unsafe"]          # index 1 = unsafe = the positive class


class SafetyClassifier:
    def __init__(self, embedder="minilm", C=1.0, seed=42):
        if embedder not in ("minilm", "tfidf"):
            raise ValueError("embedder must be 'minilm' or 'tfidf'")
        self.embedder = embedder
        self.C = C
        self.seed = seed
        self.lr = None
        self._st = None            # sentence-transformer (loaded lazily)
        self._tfidf = None
        self._cache = {}           # text -> embedding (minilm only)

    # ------------------------------------------------------------ embedding
    def _minilm(self):
        if self._st is None:
            # Imported here so tests / tfidf runs work without torch installed.
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st

    def _embed(self, texts, fit=False):
        texts = [str(t) for t in texts]
        if self.embedder == "tfidf":
            if fit:
                self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                              sublinear_tf=True, min_df=2)
                return self._tfidf.fit_transform(texts)
            return self._tfidf.transform(texts)
        # minilm: cache by text so repeated texts (e.g. plain rows) are free
        missing = sorted({t for t in texts if t not in self._cache})
        if missing:
            vecs = self._minilm().encode(missing, batch_size=64, show_progress_bar=False,
                                         convert_to_numpy=True, normalize_embeddings=True)
            self._cache.update(zip(missing, vecs))
        return np.vstack([self._cache[t] for t in texts])

    # ------------------------------------------------------------- training
    def fit(self, texts, labels):
        X = self._embed(texts, fit=True)
        y = np.array([1 if l == "unsafe" else 0 for l in labels])
        self.lr = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced",
                                     random_state=self.seed).fit(X, y)
        return self

    # ----------------------------------------------------------- prediction
    def predict_proba(self, texts):
        """P(unsafe) for each text."""
        if self.lr is None:
            raise RuntimeError("Call fit() or load() first")
        return self.lr.predict_proba(self._embed(texts))[:, 1]

    def predict(self, texts, threshold=0.5):
        return ["unsafe" if p >= threshold else "safe" for p in self.predict_proba(texts)]

    # ---------------------------------------------------------- persistence
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embedder": self.embedder, "C": self.C, "seed": self.seed,
                         "lr": self.lr, "tfidf": self._tfidf}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        clf = cls(d["embedder"], d["C"], d["seed"])
        clf.lr, clf._tfidf = d["lr"], d["tfidf"]
        return clf


@dataclass
class Verdict:
    label: str                 # 'safe' | 'unsafe'
    p_unsafe: float
    decode: DecodeResult       # what the decoder did (steps, decoded text, flags)


class VeilBreakGuard:
    """Decode first, then classify the decoded text."""

    def __init__(self, classifier):
        self.classifier = classifier

    def check(self, prompt, threshold=0.5):
        d = decode(prompt)
        # Unsafe if the decoded text OR any revealed payload looks unsafe.
        p = float(self.classifier.predict_proba(d.classify_texts).max())
        return Verdict("unsafe" if p >= threshold else "safe", p, d)
