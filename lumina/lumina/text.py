"""Shared, versioned EN/AR normalization for training, inference and matching.

Deliberately identical in behaviour to VitaMind_Journal_AI's normalizer so that a
phrase indexed by the journal analyzer and the same phrase seen by Lumina's own
models are tokenized the same way. The version string is stored in every model
config; a mismatch refuses to load rather than silently degrading.
"""
import re
import unicodedata

VERSION = "nfkc-ar-en-v1"
ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[ً-ٰٟـ]", "", text)
    text = text.translate(str.maketrans("أإآٱى", "ااااي"))
    # Retain meaningful doubles, negation and punctuation; do not conflate ة with ه.
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = re.sub(r"([ء-ي])\1{2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def language(text: str) -> str:
    return "ar" if len(ARABIC.findall(text)) > len(LATIN.findall(text)) else "en"


def clauses(text: str) -> list[str]:
    # Contrast resets scope: a denial must not erase intent stated in a later clause.
    parts = re.split(r"[.!?؟;؛\n]+|\b(?:but|however|yet)\b|(?:ولكن|لكن)", normalize(text))
    return [c.strip() for c in parts if c.strip()]
