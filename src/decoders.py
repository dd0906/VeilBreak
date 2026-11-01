"""
Decoders: undo an encoding. Pure functions, no decisions about WHETHER to
decode -- that is the detector's job (pipeline.py).

  base64   exact inverse (when the input really is Base64)
  rot13    exact inverse
  leet     approximate inverse: '4'->'a' is easy, but '1' could be 'i' OR 'l'.
           We resolve that ambiguity by asking the language model which
           spelling looks more like English (see _resolve_token).
"""

import base64
import binascii
import codecs
import itertools
import os
import re

from src.language_model import CharLM

# ---------------------------------------------------------------- Base64

# A run of 12+ Base64-alphabet characters (standard or URL-safe), optionally
# padded with '='. The look-around stops us from matching the middle of a
# longer word-like token.
B64_SPAN = re.compile(r"(?<![A-Za-z0-9+/_\-=])[A-Za-z0-9+/_\-]{12,}={0,2}(?![A-Za-z0-9+/_\-=])")


def try_base64(span):
    """
    Decode a Base64 candidate. Returns the decoded string, or None.

    We only accept it if the result is real readable text:
      - decodes as strict Base64 and valid UTF-8
      - at least 95% printable characters
      - at least 60% letters/spaces
    A random string essentially never passes all three, so this check is
    what keeps false positives near zero -- no language model needed.
    """
    s = span.strip()
    s += "=" * (-len(s) % 4)                      # repair missing padding
    try:
        if "-" in s or "_" in s:
            raw = base64.urlsafe_b64decode(s)
        else:
            raw = base64.b64decode(s, validate=True)
        text = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if len(text) < 6:
        return None
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    texty = sum(ch.isalpha() or ch.isspace() for ch in text) / len(text)
    if printable < 0.95 or texty < 0.60:
        return None
    return text


def find_base64_spans(text):
    """Return [(start, end, decoded_text)] for every readable Base64 span."""
    found = []
    for m in B64_SPAN.finditer(text):
        decoded = try_base64(m.group(0))
        if decoded is not None:
            found.append((m.start(), m.end(), decoded))
    return found


def decode_base64_spans(text):
    """
    Replace every readable Base64 span in `text` by its decoding.
    Returns (new_text, n_spans, payloads). `payloads` lists the decoded spans
    ONLY when they are embedded in a longer sentence (otherwise the whole
    text already is the payload).
    """
    spans = find_base64_spans(text)
    if not spans:
        return text, 0, []
    covered = sum(end - start for start, end, _ in spans)
    payloads = [d for _, _, d in spans] if covered < 0.9 * len(text.strip()) else []
    out, last = [], 0
    for start, end, decoded in spans:
        out.append(text[last:start])
        out.append(decoded)
        last = end
    out.append(text[last:])
    return "".join(out), len(spans), payloads


# ----------------------------------------------------------------- ROT13

def decode_rot13(text):
    return codecs.encode(text, "rot_13")


# -------------------------------------------------------------- Leetspeak
#
# Leetspeak is lossy, so decoding is a small search problem per token:
#   * '4 3 0 5 7 9 8 @ $' have one obvious meaning.
#   * '1' and '!' could be 'i' OR 'l'  -> try both, pick the better spelling
#     (a word from the vocabulary beats one that is not; ties -> language model).
#   * a symbol at the END of a token ("user$" vs "Hello!") may just be
#     punctuation -> try keeping it too.
# Every token's decoding is kept only if it looks MORE English than the
# original token, so "mp3" or "3.10" are left alone.

LEET_DIGITS = {"4": "a", "3": "e", "0": "o", "5": "s", "7": "t", "9": "g", "8": "b"}
LEET_SYMBOLS = {"@": "a", "$": "s"}
AMBIGUOUS = {"1": "il", "!": "il"}
LEET_CHARS = set(LEET_DIGITS) | set(LEET_SYMBOLS) | set(AMBIGUOUS)
SYMBOLS = set("@$!")
MAX_AMBIGUOUS = 8          # cap the 2^n search per token

_TOKEN = re.compile(r"\S+")
_VOCAB = None


def get_vocab():
    """Common English words seen in benign training prompts (tie-breaker only)."""
    global _VOCAB
    if _VOCAB is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "models", "vocab.txt")
        try:
            with open(path, encoding="utf-8") as f:
                _VOCAB = set(f.read().split())
        except OSError:
            _VOCAB = set()
    return _VOCAB


def _word(token):
    return re.sub(r"[^a-z]", "", token.lower())


def _known(word, vocab):
    """Is `word` (or its stem after dropping a common suffix) a known word?"""
    if word in vocab:
        return True
    for suffix in ("s", "es", "ed", "d", "ing", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and word[:-len(suffix)] in vocab:
            return True
    return False


def _slots(token):
    """Indexes of leet characters. A symbol counts only if touching an alphanumeric."""
    idx = []
    for i, ch in enumerate(token):
        if ch not in LEET_CHARS:
            continue
        if ch in SYMBOLS:
            prev_ok = i > 0 and token[i - 1].isalnum()
            next_ok = i + 1 < len(token) and token[i + 1].isalnum()
            if not (prev_ok or next_ok):
                continue
        idx.append(i)
    return idx


def _candidates(token, slots):
    """All plausible decodings of one token (list of strings)."""
    base = list(token)
    choices = []                                   # (position, [possible letters])
    for i in slots:
        ch = token[i]
        opts = LEET_DIGITS.get(ch) or LEET_SYMBOLS.get(ch) or AMBIGUOUS[ch]
        if ch in SYMBOLS and i == len(token) - 1:  # trailing '$' / '!' may be punctuation
            opts = opts + ch
        choices.append((i, opts))
    # cap the search: extra ambiguous slots just take their first option
    fixed = choices[MAX_AMBIGUOUS:]
    for i, opts in fixed:
        base[i] = opts[0]
    choices = choices[:MAX_AMBIGUOUS]
    out = []
    for combo in itertools.product(*[o for _, o in choices]):
        cand = base[:]
        for (i, _), letter in zip(choices, combo):
            cand[i] = letter
        out.append("".join(cand))
    return out


def _best(cands, lm, vocab):
    """
    Pick the best spelling:
      1. a real vocabulary word beats a non-word,
      2. then prefer spellings with no leftover '@ $ !' (so 'deletes' beats 'delete$'),
      3. then let the language model decide.
    """
    return max(cands, key=lambda c: (_known(_word(c), vocab),
                                     not any(ch in SYMBOLS for ch in c),
                                     lm.score(" " + c + " ")))


def decode_leetspeak(text, lm):
    """Decode leetspeak token by token. Returns (decoded_text, n_tokens_changed)."""
    vocab = get_vocab()
    pieces, last, changed = [], 0, 0
    lone = []                                       # tokens that need whole-text context
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        slots = _slots(tok)
        if not slots:
            continue
        if not any(c.isalpha() for c in tok):       # '4', '70', '!7' ... decide later
            lone.append((m, tok, slots))
            continue
        cand = _best(_candidates(tok, slots), lm, vocab)
        if lm.score(" " + cand + " ") > lm.score(" " + tok + " "):
            pieces.append((m.start(), m.end(), cand))
            changed += 1

    # Evidence rule: ONE leet-looking token ("Input1", "F1", "4m") is weak
    # evidence -- identifiers and units look like that. Require two decoded
    # tokens, or a single token that becomes a real word of 5+ letters.
    if changed < 2:
        strong = (changed == 1 and len(_word(pieces[0][2])) >= 5
                  and _known(_word(pieces[0][2]), vocab))
        if not strong:
            return text, 0

    # Second pass: leet-only tokens ('4' -> 'a', '70' -> 'to') are too ambiguous
    # on their own (could be real numbers), so we only decode them when the
    # rest of the text is clearly leetspeak AND the result is a known word.
    if changed >= 2:
        for m, tok, slots in lone:
            cands = [c for c in _candidates(tok, slots) if _known(_word(c), vocab)]
            if tok in ("4", "@"):
                cands = ["a"]
            if cands:
                pieces.append((m.start(), m.end(), cands[0]))
                changed += 1

    pieces.sort()
    out, last = [], 0
    for start, end, cand in pieces:
        out.append(text[last:start])
        out.append(cand)
        last = end
    out.append(text[last:])
    return "".join(out), changed
