"""Shared, versioned EN/AR normalization for training, inference and matching.

Deliberately identical in behaviour to journal_ai's normalizer so that a
phrase indexed by the journal analyzer and the same phrase seen by Lumina's own
models are tokenized the same way. The version string is stored in every model
config; a mismatch refuses to load rather than silently degrading.
"""
import re
import unicodedata

VERSION = "nfkc-ar-en-v2"
ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
ARABIZI_WORD = re.compile(r"(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9']+")

_ARABIZI_DIGITS = str.maketrans({"2": "ء", "3": "ع", "5": "خ", "6": "ط",
                                 "7": "ح", "8": "غ", "9": "ق"})


def arabizi_to_arabic(text: str) -> str:
    """Add a conservative Arabic spelling for Arabizi words."""
    def convert(match):
        word = match.group(0).translate(_ARABIZI_DIGITS)
        return word.replace("aa", "ا").replace("ee", "ي").replace("ii", "ي") \
            .replace("oo", "و").replace("ou", "و").replace("kh", "خ") \
            .replace("sh", "ش")
    return ARABIZI_WORD.sub(convert, text.lower())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[ً-ٰٟـ]", "", text)
    text = text.translate(str.maketrans("أإآٱىة", "اااايه"))
    # Retain meaningful doubles, negation and punctuation; do not conflate ة with ه.
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = re.sub(r"([ء-ي])\1{2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def feature_variants(text: str) -> tuple[str, ...]:
    """Return the normalized text and an Arabizi transliteration when present."""
    normalized = normalize(text)
    transliterated = normalize(arabizi_to_arabic(normalized))
    return (normalized,) if transliterated == normalized else (normalized, transliterated)


def language(text: str) -> str:
    return "ar" if len(ARABIC.findall(text)) > len(LATIN.findall(text)) else "en"


def clauses(text: str) -> list[str]:
    # Contrast resets scope: a denial must not erase intent stated in a later clause.
    parts = re.split(r"[.!?؟;؛\n]+|\b(?:but|however|yet)\b|(?:ولكن|لكن)", normalize(text))
    return [c.strip() for c in parts if c.strip()]
