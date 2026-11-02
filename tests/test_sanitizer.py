"""The sanitizer's guarantee, checked on EVERY Unicode code point (not a sample)."""
import unicodedata

import pytest

from src.sanitizer import is_invisible, sanitize

ALL_CODE_POINTS = range(0x110000)


def test_nothing_invisible_survives_any_code_point():
    # Placed between visible letters, the way an attacker would use it.
    for cp in ALL_CODE_POINTS:
        clean, _ = sanitize(f"a{chr(cp)}b")
        assert not any(is_invisible(c) for c in clean), hex(cp)


def test_every_code_point_is_removed_or_kept_never_altered():
    # Guards against a sanitizer that passes by deleting too much.
    for cp in ALL_CODE_POINTS:
        ch = chr(cp)
        clean, removed = sanitize(ch)
        if is_invisible(ch):
            assert (clean, removed) == ("", 1), hex(cp)
        elif unicodedata.category(ch) == "Zs":
            assert clean == " ", hex(cp)
        elif unicodedata.category(ch) in ("Zl", "Zp"):
            assert clean == "\n", hex(cp)
        else:
            assert (clean, removed) == (ch, 0), hex(cp)


@pytest.mark.parametrize("cp", [
    0x00AD, 0x034F, 0x061C, 0x115F, 0x17B4, 0x180E, 0x200B, 0x200C, 0x200D,
    0x200E, 0x202E, 0x2060, 0x2062, 0x2066, 0x2800, 0x3164, 0xFE0F, 0xFEFF,
    0xFFA0, 0xE0001, 0xE0041, 0xE007F, 0xE0100, 0xE01EF, 0xE000, 0xF0000, 0x0000,
    0x001B, 0x007F,
])
def test_known_invisible_characters_are_denied(cp):
    # Written out by hand so this does not simply repeat the implementation.
    assert is_invisible(chr(cp))


@pytest.mark.parametrize("text", [
    "Plain English, with punctuation! 123 $%",
    "Привет, как дела",
    "你好，世界",
    "مرحبا بالعالم",
    "line one\n\ttabbed line two",
    "emoji stay \U0001F60A\U0001F680 ❤",
])
def test_visible_text_is_untouched(text):
    assert sanitize(text) == (text, 0)


def test_unusual_spaces_become_normal_spaces():
    assert sanitize("a b c　d") == ("a b c d", 0)


def test_idempotent():
    once, _ = sanitize("a​b c\U000E0041️")
    assert sanitize(once) == (once, 0)
