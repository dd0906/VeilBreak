"""
Decoders: undo an encoding. Pure functions, no decisions about WHETHER to
decode -- that is the detector's job (pipeline.py).

  base64   exact inverse (when the input really is Base64)
  rot13    exact inverse
  hex      exact inverse (when the input really is hex of readable text)
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


# ------------------------------------------------------------------- Hex

# An even-length run of 12+ hex characters. A hash or random id also matches this,
# which is why try_hex only accepts a result that is real readable text.
HEX_SPAN = re.compile(r"(?<![0-9a-zA-Z])(?:[0-9a-fA-F]{2}){6,}(?![0-9a-zA-Z])")


def try_hex(span):
    """Decode a hex candidate. Returns the decoded string, or None (same checks as Base64)."""
    try:
        text = bytes.fromhex(span).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    texty = sum(ch.isalpha() or ch.isspace() for ch in text) / len(text)
    return text if printable >= 0.95 and texty >= 0.60 else None


def decode_hex_spans(text):
    """Replace every readable hex span by its decoding. Returns (new_text, n_spans, payloads)."""
    spans = []
    for m in HEX_SPAN.finditer(text):
        decoded = try_hex(m.group(0))
        if decoded is not None:
            spans.append((m.start(), m.end(), decoded))
    return _with_payloads(text, spans) if spans else (text, 0, [])


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


# ---------------------------------------------------- Invisible characters
#
# Current "smuggling" attacks hide text in Unicode characters that a person
# cannot see but an LLM can still read. All four are exact or near-exact to
# undo, so they are plain rules with no language model:
#   tag_smuggling    letters copied into the invisible Tags block (U+E00xx)
#   emoji_smuggling  bytes hidden in variation selectors after an emoji
#   zero_width       invisible characters inserted inside words
#   homoglyph        Latin letters swapped for look-alike Cyrillic/Greek ones

TAG_RUN = re.compile("[\U000E0020-\U000E007E]+")
TAG_OFFSET = 0xE0000
BLACK_FLAG = "\U0001F3F4"      # subdivision flags (Scotland...) legitimately use tags

VS_RUN = re.compile("[︀-️\U000E0100-\U000E01EF]{4,}")   # one alone is a normal emoji style

ZERO_WIDTH_RUN = re.compile("[​‌‍⁠﻿­]+")
JOINERS = set("‌‍")  # also used by real emoji and some scripts (see strip_zero_width)

# Look-alike letter -> the Latin letter it imitates.
HOMOGLYPHS = {
    # Cyrillic
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
    # Greek
    "ο": "o", "α": "a", "ν": "v", "ρ": "p", "ι": "i",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}


def _with_payloads(text, spans):
    """Replace (start, end, hidden) spans. A payload is listed separately only
    if something visible around it (words, or even a lone emoji) could dilute it."""
    out, last = [], 0
    for start, end, hidden in spans:
        out.append(text[last:start])
        out.append(hidden)
        last = end
    out.append(text[last:])
    visible = "".join(text[a:b] for a, b in zip([0] + [e for _, e, _ in spans],
                                                [s for s, _, _ in spans] + [len(text)]))
    payloads = [h for _, _, h in spans] if visible.strip() else []
    return "".join(out), len(spans), payloads


def decode_tag_smuggling(text):
    """Reveal text hidden in Unicode Tag characters. Returns (text, n_spans, payloads)."""
    spans = [(m.start(), m.end(), "".join(chr(ord(c) - TAG_OFFSET) for c in m.group(0)))
             for m in TAG_RUN.finditer(text)
             if not (m.start() > 0 and text[m.start() - 1] == BLACK_FLAG)]
    return _with_payloads(text, spans) if spans else (text, 0, [])


def _vs_to_byte(ch):
    cp = ord(ch)
    return cp - 0xFE00 if cp <= 0xFE0F else cp - 0xE0100 + 16


def decode_emoji_smuggling(text):
    """Reveal bytes hidden in variation selectors. Returns (text, n_spans, payloads)."""
    spans = []
    for m in VS_RUN.finditer(text):
        try:
            hidden = bytes(_vs_to_byte(c) for c in m.group(0)).decode("utf-8")
        except UnicodeDecodeError:
            continue
        if all(c.isprintable() or c in "\n\t" for c in hidden):
            spans.append((m.start(), m.end(), hidden))
    return _with_payloads(text, spans) if spans else (text, 0, [])


def strip_zero_width(text):
    """Remove invisible characters inserted into words. Returns (text, n_removed).

    Joiners are kept unless they follow an ASCII letter or digit: they are how
    real emoji sequences and some non-Latin scripts are written.
    """
    removed = 0

    def repl(m):
        nonlocal removed
        run = m.group(0)
        before = text[m.start() - 1] if m.start() > 0 else ""
        if set(run) <= JOINERS and not (before.isascii() and before.isalnum()):
            return run
        removed += len(run)
        return ""

    return ZERO_WIDTH_RUN.sub(repl, text), removed


def _is_foreign_letter(ch):
    return ch.isalpha() and not ch.isascii() and ch not in HOMOGLYPHS


def decode_homoglyphs(text):
    """Map look-alike letters back to Latin. Returns (text, n_replaced).

    Only fires with evidence of a swap: at least one word mixes Latin letters
    with look-alikes. Real Russian or Greek text, or a lone 'α' in a maths
    question, has no such word and passes through untouched.
    """
    mixed = False
    for tok in _TOKEN.findall(text):
        letters = [c for c in tok if c.isalpha()]
        has_latin = any(c.isascii() for c in letters)
        has_lookalike = any(c in HOMOGLYPHS for c in letters)
        if has_latin and has_lookalike and not any(_is_foreign_letter(c) for c in letters):
            mixed = True
            break
    if not mixed:
        return text, 0
    n = sum(c in HOMOGLYPHS for c in text)
    return "".join(HOMOGLYPHS.get(c, c) for c in text), n
