"""
Default-deny Unicode sanitizer: the last gate before text reaches a model.

decoders.py REVEALS hidden text it knows how to read. This module handles
everything else: any invisible character still present is removed, whether or
not we recognise the scheme using it.

The guarantee, which tests/test_sanitizer.py checks on every Unicode code point:
    sanitize() output contains no character for which is_invisible() is True.

"Invisible" has a precise meaning here, so the guarantee is testable:
  * Unicode's own Default_Ignorable_Code_Point list (zero-width characters,
    tag characters, variation selectors, fillers...)
  * categories Cf, Cc, Co, Cs, Cn (format, control, private use, surrogate,
    unassigned), except newline and tab
  * the blank Braille pattern U+2800, which draws nothing

Deliberately no emoji exceptions. Joiners and variation selectors between emoji
could carry hidden bits, so allowing "valid" sequences would leave a channel
open. The trade-offs, accepted because this output is only read by a model:
  * a family emoji becomes three separate emoji
  * subdivision flags (Scotland...) become a plain black flag
  * Persian and some Indic words lose their joiners, though they stay readable

What this does NOT promise: that a prompt is safe. Look-alike letters are visible
characters, and spaces or combining marks can still carry data. Those are
out of scope for this guarantee.
"""
import unicodedata

# Default_Ignorable_Code_Point, from Unicode's DerivedCoreProperties.txt.
DEFAULT_IGNORABLE = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160),
    (0x17B4, 0x17B5), (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E),
    (0x2060, 0x206F), (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x1BCA0, 0x1BCA3), (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
DENIED_CATEGORIES = {"Cf", "Cc", "Co", "Cs", "Cn"}
BLANK_BRAILLE = 0x2800
KEEP = {"\n", "\t"}


def is_invisible(ch):
    """Is this character on the deny list? (See the module docstring.)"""
    if ch in KEEP:
        return False
    cp = ord(ch)
    if cp == BLANK_BRAILLE or any(lo <= cp <= hi for lo, hi in DEFAULT_IGNORABLE):
        return True
    return unicodedata.category(ch) in DENIED_CATEGORIES


def sanitize(text):
    """Return (clean_text, n_removed). Unusual spaces become a normal space and
    line/paragraph separators become a newline: they are not counted as removed."""
    out, removed = [], 0
    for ch in text:
        if is_invisible(ch):
            removed += 1
            continue
        category = unicodedata.category(ch)
        if category == "Zs":
            out.append(" ")
        elif category in ("Zl", "Zp"):
            out.append("\n")
        else:
            out.append(ch)
    return "".join(out), removed
