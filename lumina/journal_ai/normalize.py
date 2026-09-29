"""Shared, versioned normalization for training, inference and phrase matching."""
import re
import unicodedata

VERSION = "nfkc-ar-en-v1"
ARABIC = re.compile(r"[\u0600-\u06ff]")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    text = text.translate(str.maketrans("أإآٱى", "ااااي"))
    # Retain meaningful doubles, negation and punctuation; do not conflate ة with ه.
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = re.sub(r"([\u0621-\u064a])\1{2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def language(text: str) -> str:
    return "ar" if len(ARABIC.findall(text)) > len(re.findall(r"[A-Za-z]", text)) else "en"

def clauses(text: str) -> list[str]:
    # Contrast resets scope: a denial must not erase intent in a later clause.
    return [c.strip() for c in re.split(r"[.!?؟;؛\n]+|\b(?:but|however|yet)\b|(?:ولكن|لكن|بس الحين)", normalize(text)) if c.strip()]

def contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(normalize(phrase)) + r"(?!\w)", normalize(text)))
