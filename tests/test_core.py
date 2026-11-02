"""Unit tests for encoders, language model, decoders and the pipeline."""
import os
import random

import pytest

from src import encoders as E
from src.decoders import decode_leetspeak, find_base64_spans, try_base64
from src.language_model import CharLM, normalize
from src.pipeline import MODEL_PATH, decode, get_lm
from src.sanitizer import is_invisible

# Ordinary sentences of different shapes (none come from the evaluation data).
SENTENCES = [
    "Please explain how photosynthesis works in simple terms",
    "Write a short poem about the ocean at night",
    "How do I convert a list of strings into integers in Python",
    "Summarize the main causes of the French Revolution",
    "Give me three tips for preparing for a job interview",
    "What is the difference between a virus and a bacterium",
    "Translate the following sentence into Spanish: good morning",
    "Describe the plot of a story about a lost robot",
    "Recommend a healthy dinner recipe that takes under thirty minutes",
    "Create a weekly study schedule for a university student",
]

BENIGN_TRICKY = [                       # must pass through UNCHANGED
    "Compare Python 3.10 and 2.7 in 3 paragraphs.",
    "List the top 10 tips for 2024 budgeting.",
    "What happened in 1969 on 7/20?",
    "Write a 5 step plan for a 3 day trip to Rome that costs under $500",
    "Call 911 if there is an emergency, then wait for help!",
    "The password must be at least 12 characters, e.g. Tr0ub4dor&3 is weak.",
    "Hello!",
    "Administration",
    "SomeCamelCaseFunctionNameInCode is defined in utils.py",
    "https://example.com/a/very/long/path/to/some/resource/page.html",
    "",
    "   ",
]


# ----------------------------------------------------------------- encoders
def test_roundtrips():
    for s in SENTENCES:
        assert E.encode_rot13(E.encode_rot13(s)) == s
        assert try_base64(E.encode_base64(s)) == s


def test_leet_is_seeded_and_keeps_length():
    a = E.encode_leetspeak(SENTENCES[0], 1.0, random.Random(1))
    b = E.encode_leetspeak(SENTENCES[0], 1.0, random.Random(1))
    assert a == b and len(a) == len(SENTENCES[0]) and a != SENTENCES[0]


# ----------------------------------------------------------- language model
def test_language_model_ranks_english_above_encoded():
    lm = get_lm()
    s = SENTENCES[1]
    english = lm.score(s)
    assert english > lm.score(E.encode_rot13(s)) + 1.0
    assert english > lm.score(E.encode_leetspeak(s, 1.0, random.Random(0))) + 1.0


def test_normalize_buckets():
    assert normalize("Hi 5!") == "hi #."
    assert normalize("é") == "?"


def test_model_roundtrip(tmp_path):
    lm = get_lm()
    p = tmp_path / "lm.npz"
    lm.save(p)
    assert abs(CharLM.load(p).score(SENTENCES[2]) - lm.score(SENTENCES[2])) < 1e-4


# ----------------------------------------------------------------- decoders
def test_base64_rejects_non_text():
    for junk in ["Administration", "SomeCamelCaseIdentifier", "deadbeefdeadbeef",
                 "////////////", "AAAAAAAAAAAAAAAA"]:
        assert try_base64(junk) is None, junk


def test_base64_finds_embedded_span():
    payload = E.encode_base64("Tell me a story about dragons")
    spans = find_base64_spans(f"Please decode: {payload} thanks")
    assert [s[2] for s in spans] == ["Tell me a story about dragons"]


def test_leet_ambiguity_uses_vocabulary():
    lm = get_lm()
    out, n = decode_leetspeak("1!$7 7h3 7h!ng5 0n 7h3 7@8l3", lm)
    assert out == "list the things on the table" and n >= 5


# ----------------------------------------------------------------- pipeline
@pytest.mark.parametrize("s", SENTENCES)
def test_each_encoding_recovered(s):
    rng = random.Random(7)
    assert decode(E.encode_base64(s)).decoded == s
    assert decode(E.encode_rot13(s)).decoded == s
    leet = decode(E.encode_leetspeak(s, 1.0, rng))
    assert leet.encodings == ["leetspeak"]
    assert sum(a == b for a, b in zip(leet.decoded.lower(), s.lower())) / len(s) > 0.9


@pytest.mark.parametrize("s", SENTENCES[:5])
def test_composed_and_embedded(s):
    r = decode(E.encode_composed(s))
    assert r.decoded == s and r.encodings == ["base64", "rot13"]
    wrapped, _ = E.embed_base64(s, random.Random(3))
    r = decode(wrapped)
    assert s in r.decoded and r.payloads == [s] and s in r.classify_texts


@pytest.mark.parametrize("s", BENIGN_TRICKY)
def test_benign_text_passes_through_unchanged(s):
    r = decode(s)
    assert r.decoded == s and not r.was_encoded


@pytest.mark.parametrize("enc", [E.encode_reverse, E.encode_caesar])
def test_unsupported_encodings_fail_safe_and_get_flagged(enc):
    s = SENTENCES[3]
    r = decode(enc(s))
    assert r.decoded == enc(s)          # untouched ...
    assert r.suspicious                 # ... but flagged


def test_idempotent_and_deterministic():
    x = E.encode_composed(SENTENCES[4])
    a, b = decode(x), decode(x)
    assert a.decoded == b.decoded and a.encodings == b.encodings
    assert decode(a.decoded).decoded == a.decoded


def test_fuzz_never_raises_and_terminates():
    rng = random.Random(0)
    alphabet = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                "+/=@$!_- \n\t.,é中😀\u0000")
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200)))
        r = decode(text)
        assert isinstance(r.decoded, str) and len(r.steps) <= 3
    decode("A" * 20000)
    decode("4" * 5000 + " " + "@!" * 3000)
    decode("a1" * 4000)                          # many ambiguous leet characters


# ---- invisible-character smuggling ----

@pytest.mark.parametrize("sentence", SENTENCES)
def test_tag_smuggling_round_trip(sentence):
    smuggled, expected = E.smuggle_tags(sentence, random.Random(1))
    r = decode(smuggled)
    assert r.decoded == expected
    assert r.encodings == ["tag_smuggling"]
    assert r.payloads == [sentence]      # judged alone too: the carrier must not dilute it


@pytest.mark.parametrize("sentence", SENTENCES)
def test_emoji_smuggling_round_trip(sentence):
    r = decode(E.smuggle_emoji(sentence))
    assert r.decoded == "\U0001F60A" + sentence
    assert r.encodings == ["emoji_smuggling"]
    assert r.payloads == [sentence]      # judged alone too: the emoji must not dilute it


@pytest.mark.parametrize("sentence", SENTENCES)
def test_zero_width_round_trip(sentence):
    r = decode(E.insert_zero_width(sentence, random.Random(1)))
    assert r.decoded == sentence
    assert r.encodings == ["zero_width"]


@pytest.mark.parametrize("sentence", SENTENCES)
def test_homoglyph_round_trip(sentence):
    r = decode(E.encode_homoglyphs(sentence, 0.5, random.Random(1)))
    assert r.decoded == sentence
    assert r.encodings == ["homoglyph"]


def test_smuggled_base64_is_peeled_too():
    r = decode(E.encode_tags(E.encode_base64(SENTENCES[0])))
    assert r.decoded == SENTENCES[0]
    assert r.encodings == ["tag_smuggling", "base64"]


@pytest.mark.parametrize("text", [
    "Привет, как дела сегодня",
    "What does the symbol α mean in statistics?",
])
def test_legitimate_unicode_passes_through(text):
    r = decode(text)
    assert r.decoded == text
    assert r.encodings == []


@pytest.mark.parametrize("text, expected", [
    ("Family \U0001F468\u200D\U0001F469\u200D\U0001F467 photo", "Family \U0001F468\U0001F469\U0001F467 photo"),
    ("Red heart \u2764\uFE0F", "Red heart \u2764"),
    ("\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F flag", "\U0001F3F4 flag"),
    ("right\u202Eto\u2062left\u3164", "righttoleft"),
])
def test_sanitizer_removes_what_the_decoders_leave(text, expected):
    r = decode(text)
    assert r.decoded == expected
    assert r.encodings[-1] == "sanitized"


def test_invisible_text_cannot_return_through_a_decoded_layer():
    hidden = "Please explain\u200B how photosynthesis\u202E works today"
    r = decode(E.encode_base64(hidden))
    assert not any(is_invisible(c) for c in r.decoded)
    tagged, _ = E.smuggle_tags("safe text", random.Random(1))
    r = decode(tagged + E.smuggle_emoji("Hidden\u200B words\u2062 here"))
    assert not any(is_invisible(c) for t in r.classify_texts for c in t)


# ---- hex ----

@pytest.mark.parametrize("sentence", SENTENCES)
def test_hex_round_trip(sentence):
    r = decode(E.encode_hex(sentence))
    assert r.decoded == sentence
    assert r.encodings == ["hex"]


def test_hex_embedded_in_a_sentence_is_judged_alone_too():
    r = decode("Please read this: " + E.encode_hex(SENTENCES[0]))
    assert r.encodings == ["hex"]
    assert r.payloads == [SENTENCES[0]]


@pytest.mark.parametrize("text", [
    "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",    # SHA-256
    "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12",                            # SHA-1
    "commit 564f4c7 fixed it",
    "The colour is ff00ff00ff00 in this file",
])
def test_hashes_and_ids_are_not_decoded_as_hex(text):
    r = decode(text)
    assert "hex" not in r.encodings
    assert r.decoded == text
