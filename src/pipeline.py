"""
The VeilBreak decode pipeline.

    text --> [ try each decoder ] --> accept the best one --> repeat (max 3x)
                                                  |
                                       nothing helps? stop, return text as-is

A decoder step is ACCEPTED only with evidence:
  * base64    : the span decodes to readable text (structural check)
  * hex       : same structural check, so hashes and random ids stay untouched
  * rot13     : the result looks clearly MORE English than the input
  * leetspeak : same, checked token by token
Because every accepted step must make the text more readable (or reveal a
valid Base64 payload), the loop always terminates and unknown / plain text
passes through UNCHANGED -- the safe fallback.

Before that loop, one pass removes invisible-character smuggling (tag
characters, emoji variation selectors, zero-width characters, homoglyphs). It
runs first because those tricks can wrap any of the encodings above, e.g. a
Base64 string hidden in tag characters. The sanitizer (sanitizer.py) then runs
last on everything we return, so a decoded layer cannot bring invisible
characters back in.

Design trade-off: we use rules + a small language model instead of a trained
detector. It is deterministic and explainable, but it only knows the encodings
it was built for. For anything else it cannot decode, so it raises the
`suspicious` flag (text looks like gibberish) and leaves the decision to the
caller.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

from src.decoders import (decode_base64_spans, decode_emoji_smuggling,
                          decode_hex_spans, decode_homoglyphs,
                          decode_leetspeak, decode_rot13,
                          decode_tag_smuggling, find_base64_spans,
                          strip_zero_width)
from src.language_model import CharLM
from src.sanitizer import sanitize

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "models", "char_lm.npz")

# ---- Thresholds (average log-prob per character; see docs/METHODOLOGY.md) ----
# These were chosen on a development split, not on the evaluation set.
ROT13_MIN_GAIN = 0.60       # decoded must beat the original by this much
ROT13_MIN_SCORE = -3.00     # ...and itself look like English
ROT13_MIN_LETTERS = 12      # too little text -> too noisy -> don't guess
SUSPICIOUS_BELOW = -3.40    # unreadable AND undecodable -> flag it
SUSPICIOUS_MIN_CHARS = 20
MAX_DEPTH = 3               # how many layers of encoding we will peel


@dataclass
class Step:
    transform: str          # 'base64' | 'hex' | 'rot13' | 'leetspeak' | an invisible-character step
    detail: str = ""        # e.g. "2 spans" -- for the demo / debugging
    gain: float = 0.0       # LM score improvement (0 for base64)
    payloads: List[str] = field(default_factory=list)  # decoded spans embedded in a sentence


@dataclass
class DecodeResult:
    original: str
    decoded: str
    steps: List[Step] = field(default_factory=list)
    suspicious: bool = False   # looks like gibberish we could not decode

    @property
    def payloads(self):
        """Hidden messages revealed inside a longer sentence (e.g. 'Decode this: <b64>')."""
        return [p for s in self.steps for p in s.payloads]

    @property
    def classify_texts(self):
        """What the safety classifier should look at: the decoded text AND each
        revealed payload on its own (a harmless wrapper sentence must not dilute
        a harmful payload)."""
        return [self.decoded] + [p for p in self.payloads if p != self.decoded]

    @property
    def was_encoded(self):
        return bool(self.steps)

    @property
    def encodings(self):
        return [s.transform for s in self.steps]


_lm = None


def get_lm():
    """Load the language model once (lazy, cached)."""
    global _lm
    if _lm is None:
        _lm = CharLM.load(MODEL_PATH)
    return _lm


def _letters(text):
    return sum(c.isalpha() for c in text)


def _candidates(text, lm, last):
    """Every decoder that could apply right now -> list of (Step, new_text, rank).
    `last` = the transform applied in the previous step (rot13/leetspeak are not
    allowed twice in a row: rot13 twice undoes itself, leetspeak is one-pass)."""
    cands = []
    base = lm.score(text)

    # 1) Base64 (whole string or embedded). Structural check, no LM needed.
    new, n, payloads = decode_base64_spans(text)
    if n:
        cands.append((Step("base64", f"{n} span(s)", payloads=payloads), new, 10.0))

    # Hex (whole string or embedded). Same structural check as Base64.
    new, n, payloads = decode_hex_spans(text)
    if n:
        cands.append((Step("hex", f"{n} span(s)", payloads=payloads), new, 9.0))

    # 2) ROT13 (whole text). Needs a clear language-model win.
    if last != "rot13" and _letters(text) >= ROT13_MIN_LETTERS:
        new = decode_rot13(text)
        gain = lm.score(new) - base
        if gain >= ROT13_MIN_GAIN and lm.score(new) >= ROT13_MIN_SCORE:
            cands.append((Step("rot13", gain=gain), new, gain))
        elif find_base64_spans(new):
            # ROT13 wrapped around a Base64 payload: the LM can't see the
            # win, but the result contains a readable Base64 span.
            cands.append((Step("rot13", "reveals base64", gain=0.0), new, 0.5))

    # 3) Leetspeak (token by token, each token must look more English).
    new, n = decode_leetspeak(text, lm) if last != "leetspeak" else (text, 0)
    if n:
        gain = lm.score(new) - base
        if gain > 0:
            cands.append((Step("leetspeak", f"{n} token(s)", gain=gain), new, gain))

    return cands


def _reveal_invisible(text):
    """Undo invisible-character smuggling. Returns (new_text, [Step, ...]).
    Each trick is checked once, in this order; hidden text is revealed before
    the look-alike rules run on it."""
    steps = []
    new, n, payloads = decode_tag_smuggling(text)
    if n:
        steps.append(Step("tag_smuggling", f"{n} span(s)", payloads=payloads))
    text = new
    new, n, payloads = decode_emoji_smuggling(text)
    if n:
        steps.append(Step("emoji_smuggling", f"{n} span(s)", payloads=payloads))
    text = new
    new, n = strip_zero_width(text)
    if n:
        steps.append(Step("zero_width", f"{n} char(s)"))
    text = new
    new, n = decode_homoglyphs(text)
    if n:
        steps.append(Step("homoglyph", f"{n} char(s)"))
    return new, steps


_HEX_ONLY = re.compile(r"[0-9a-fA-F]{20,}")


def _looks_undecodable(text, lm):
    """Gibberish we could not decode: hex blob, or text far from English."""
    t = text.strip()
    if _HEX_ONLY.fullmatch(t):
        return True
    return len(t) >= SUSPICIOUS_MIN_CHARS and lm.score(t) < SUSPICIOUS_BELOW


def _sanitize_result(current, steps):
    """Remove leftover invisible characters from the text and every payload."""
    clean, removed = sanitize(current)
    for step in steps:
        step.payloads = [sanitize(p)[0] for p in step.payloads]
    if removed:
        steps.append(Step("sanitized", f"{removed} char(s)"))
    return clean, steps


def decode(text, max_depth=MAX_DEPTH):
    """Peel off up to `max_depth` encoding layers. Never raises."""
    lm = get_lm()
    try:
        current, steps = _reveal_invisible(text)
        for _ in range(max_depth):
            cands = _candidates(current, lm, steps[-1].transform if steps else None)
            if not cands:
                break
            step, new_text, _rank = max(cands, key=lambda c: c[2])
            if new_text == current:
                break
            steps.append(step)
            current = new_text
        suspicious = (not steps and _looks_undecodable(current, lm))
        current, steps = _sanitize_result(current, steps)
        return DecodeResult(text, current, steps, suspicious)
    except Exception:                      # safe fallback: pass text through, but still clean
        clean, steps = _sanitize_result(text, [])
        return DecodeResult(text, clean, steps, False)


def decode_many(texts):
    """Convenience: decode a list of prompts."""
    return [decode(t) for t in texts]
