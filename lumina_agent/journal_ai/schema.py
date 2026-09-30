from dataclasses import asdict, dataclass, field
from typing import Literal

TIERS = ("none", "low", "moderate", "moderate_flagged", "high")
TRAIN_TIERS = ("none", "low", "moderate", "high")
CATEGORIES = ("suicidal_ideation", "command_hallucination", "paranoia", "hopelessness", "anxiety", "sadness", "overload", "elevated", "idiom")
RANK = {tier: i for i, tier in enumerate(TIERS)}
MAX_TEXT_LENGTH = 12000

@dataclass
class Context:
    tier: str = "none"
    categories: list[str] = field(default_factory=list)
    subject: str = "self"
    temporal: str = "current"
    negated: bool = False
    is_idiom: bool = False
    confidence: float = 0.0
    rationale: str = ""
    source: str = "rules"
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text exceeds {MAX_TEXT_LENGTH} characters; split the entry explicitly")
    if "\x00" in text:
        raise ValueError("text contains a NUL character")
    return text
