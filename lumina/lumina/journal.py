"""Journal AI as a module inside the Lumina agent.

`journal_ai` is an *analyzer*, not a second assistant (spec s44): entry
text in, structured signals out. It never persists, never replies to the patient
and never decides anything. This module is the boundary that holds it to that,
and the contract Lumina core consumes.

    JournalEntry text
        -> JournalSentinel (lexicon + its own classifier + clause scoping)
        -> JournalAnalysisResult   (this module's contract, spec s45)
        -> Lumina: state signals, safety fusion, memory *candidates*

Two rules from the spec shape the output.

* Raw journal text never leaves this module (s46). The result carries a neutral
  structural summary and numeric signals - not sentences the patient wrote. Chat
  turns therefore never receive journal prose, only signals.
* Nothing from a journal becomes a fact. Every memory suggestion is a
  *candidate* with a confidence and a justification, and the memory store's own
  floor decides whether it is ever promoted (s41, s46).

Journal text is untrusted input (s114, s115). It is only ever pattern-matched and
classified; no part of it is interpreted as an instruction to the system.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

from .state import SCALE_MAX
from .taxonomy import SAFETY_ORDER

ANALYSIS_VERSION = "lumina-journal-analysis-v1"

# Journal cue category -> the state dimensions it speaks to, and which way.
#
# These are *estimates read off text*, which is why build_state marks them
# `estimated` and lets any check-in value override them. The magnitudes are
# deliberately modest: a journal mentioning stress is evidence of stress, not a
# measurement of it.
CATEGORY_SIGNALS = {
    "anxiety": {"stress": +2.0},
    "overload": {"stress": +2.5, "focus": -2.0, "task_completion": -2.0},
    "sadness": {"mood": -2.5, "energy": -1.5},
    "hopelessness": {"mood": -3.5, "energy": -2.0},
    "elevated": {"energy": +3.0, "mood": +1.5},
    "paranoia": {"stress": +2.5, "social_connection": -2.0},
    "suicidal_ideation": {"mood": -3.0},
    "command_hallucination": {"stress": +2.5},
    # An idiom carries no state information at all.
    "idiom": {},
}

# Where the midpoint sits before a category moves it. Anything untouched by a
# category is simply absent from the result rather than defaulted.
NEUTRAL = SCALE_MAX / 2

# Journal categories that justify proposing a durable memory candidate.
MEMORY_WORTHY = {
    "hopelessness": ("PATTERN", "Journal entries have expressed hopelessness."),
    "paranoia": ("CONTEXT", "Journal entries have described feeling watched or targeted."),
    "overload": ("PATTERN", "Journal entries have described being overloaded by tasks."),
    "elevated": ("PATTERN", "Journal entries have described unusually elevated energy."),
}

# Category -> what a follow-up should be about.
FOLLOW_UP = {
    "overload": "task_initiation",
    "anxiety": "stress_reduction",
    "sadness": "mood",
    "hopelessness": "safety_check",
    "paranoia": "grounding",
    "elevated": "sleep_consistency",
    "suicidal_ideation": "safety_check",
    "command_hallucination": "grounding",
}

TIER_TO_LEVEL = {"none": "NORMAL", "low": "NORMAL", "moderate": "ELEVATED",
                 "moderate_flagged": "ELEVATED", "high": "CRISIS"}


@dataclass
class MemoryCandidate:
    category: str
    key: str
    content: str
    confidence: float
    justification: str
    source: str = "MODEL_INFERENCE"
    status: str = "CANDIDATE"

    def to_dict(self):
        return asdict(self)


@dataclass
class JournalAnalysisResult:
    """The contract Lumina core consumes. Carries no raw journal text."""
    entry_id: str | None
    language: str
    summary: str
    signals: dict = field(default_factory=dict)
    topics: list = field(default_factory=list)
    emotion: dict | None = None
    safety: dict = field(default_factory=dict)
    memory_candidates: list = field(default_factory=list)
    follow_up_candidates: list = field(default_factory=list)
    uncertainty: list = field(default_factory=list)
    model: dict = field(default_factory=dict)
    content_version: int = 1
    analysis_version: str = ANALYSIS_VERSION
    analyzed_at: str = ""
    contains_raw_text: bool = False
    is_diagnostic: bool = False

    @property
    def idempotency_key(self):
        """Stable key so a retried job cannot double-write (spec s90)."""
        return f"journal:{self.entry_id}:{self.content_version}:{self.analysis_version}"

    def to_dict(self):
        data = asdict(self)
        data["idempotency_key"] = self.idempotency_key
        return data


class JournalAnalyzer:
    """Wraps journal_ai behind the Lumina contract."""

    def __init__(self, sentinel=None, emotion_model=None):
        if sentinel is None:
            from journal_ai.pipeline import JournalSentinel
            sentinel = JournalSentinel()
        self.sentinel = sentinel
        # Optional. The trained emotion head is weak on distress emotions, so it
        # is consulted but its abstention is respected and reported.
        self.emotion_model = emotion_model

    def analyze(self, text, *, entry_id=None, language=None, content_version=1,
                is_private=False, analysis_consent=True, now=None):
        """Analyze one entry. Returns a JournalAnalysisResult, writes nothing.

        `is_private` / `analysis_consent` mirror the product rules: a private
        entry may only be analyzed for the patient's own support when consent to
        journal analysis exists. Enforcement is the backend's, but refusing here
        too means a mis-wired caller cannot quietly bypass it.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("journal text must be a non-empty string")
        if is_private and not analysis_consent:
            raise PermissionError(
                "private entry cannot be analyzed without journal analysis consent")

        raw = self.sentinel.analyze(text, lang=language, entry_id=entry_id,
                                   now=now, include_audit=True)
        audit = raw.get("audit", {})
        categories = list(raw.get("categories", []))
        tier = raw["tier"]

        signals = self._signals(categories)
        uncertainty = self._uncertainty(raw, audit, categories)

        return JournalAnalysisResult(
            entry_id=entry_id,
            language=raw.get("language") or language or "en",
            summary=self._summary(raw, categories),
            signals=signals,
            topics=sorted(categories),
            emotion=self._emotion(text),
            safety={
                "level": TIER_TO_LEVEL.get(tier, "ELEVATED"),
                "journal_tier": tier,
                "flags": sorted(categories),
                "subject": raw.get("subject"),
                "temporal": raw.get("temporal"),
                "negated": raw.get("negated"),
                "is_idiom": raw.get("is_idiom"),
                "reason": audit.get("fusion_reason"),
                "requires_human_review": raw.get("review", {}).get("requires_human_review", True),
            },
            memory_candidates=[c.to_dict() for c in self._memory(categories, raw)],
            follow_up_candidates=sorted(
                {FOLLOW_UP[c] for c in categories if c in FOLLOW_UP}),
            uncertainty=uncertainty,
            model={
                "name": "vitamind_journal_ai",
                "version": raw.get("model", {}).get("version"),
                "available": raw.get("model", {}).get("available"),
                "source": raw.get("model", {}).get("source"),
                "lexicon_version": audit.get("lexicon_version"),
                "clinically_validated": False,
            },
            content_version=content_version,
            analyzed_at=raw.get("analyzed_at")
                        or dt.datetime.now(dt.timezone.utc).isoformat(),
        )

    # -- pieces -----------------------------------------------------------
    def _signals(self, categories):
        """Turn fired categories into 0-10 state estimates.

        A dimension no category touched is left out entirely, so build_state marks
        it `missing` rather than inventing a neutral reading.
        """
        deltas = {}
        for category in categories:
            for dimension, delta in CATEGORY_SIGNALS.get(category, {}).items():
                # Keep the strongest single push per dimension rather than summing,
                # so three overlapping cues cannot drive a value to the rail.
                if abs(delta) > abs(deltas.get(dimension, 0.0)):
                    deltas[dimension] = delta
        return {dimension: round(max(0.0, min(SCALE_MAX, NEUTRAL + delta)), 2)
                for dimension, delta in deltas.items()}

    def _summary(self, raw, categories):
        """A structural description. Never a quote, never a paraphrase of content."""
        if not categories:
            return "Entry recorded; no support-relevant cues detected."
        subject = raw.get("subject", "self")
        temporal = raw.get("temporal", "current")
        listed = ", ".join(sorted(categories))
        scope = []
        if subject == "other":
            scope.append("attributed to another person")
        if temporal == "past":
            scope.append("described as past")
        if raw.get("negated"):
            scope.append("negated")
        if raw.get("is_idiom"):
            scope.append("idiomatic")
        suffix = f" ({'; '.join(scope)})" if scope else ""
        return f"Entry contains cues for: {listed}{suffix}."

    def _emotion(self, text):
        if self.emotion_model is None:
            return None
        try:
            prediction = self.emotion_model.predict(text)["emotion"]
        except Exception:
            return None
        return {"label": prediction["label"],
                "confidence": prediction["confidence"],
                "abstained": prediction["abstain"],
                "reliable": False,
                "note": "emotion head is weak on distress classes; treat as a hint"}

    def _memory(self, categories, raw):
        """Propose candidates only. Nothing from a journal becomes a fact."""
        # Attributed, past, negated or idiomatic cues describe something other
        # than the patient's present state and must not seed a memory.
        if raw.get("subject") == "other" or raw.get("temporal") == "past" \
                or raw.get("negated") or raw.get("is_idiom"):
            return []
        confidence = raw.get("model", {}).get("confidence")
        candidates = []
        for category in sorted(categories):
            if category not in MEMORY_WORTHY:
                continue
            memory_category, content = MEMORY_WORTHY[category]
            candidates.append(MemoryCandidate(
                category=memory_category,
                key=f"journal_{category}",
                content=content,
                # Deliberately below the memory store's promotion floor: a single
                # entry is a hint, and the store keeps it as a candidate.
                confidence=round(min(0.7, float(confidence or 0.5)), 3),
                justification=f"journal cue {category!r} in a current first-person entry",
            ))
        return candidates

    def _uncertainty(self, raw, audit, categories):
        reasons = []
        if not raw.get("model", {}).get("available"):
            reasons.append("journal_classifier_unavailable")
        if audit.get("model_error_type"):
            reasons.append(f"journal_model_error:{audit['model_error_type']}")
        if raw.get("tier") == "moderate_flagged":
            reasons.append("flagged_for_review_by_rules")
        if raw.get("is_idiom") and categories:
            reasons.append("idiomatic_language_present")
        if raw.get("subject") == "other":
            reasons.append("cues_attributed_to_another_person")
        if raw.get("temporal") == "past":
            reasons.append("cues_described_as_past")
        confidence = raw.get("model", {}).get("confidence")
        if confidence is not None and confidence < 0.65:
            reasons.append("low_classifier_confidence")
        return reasons


def fuse_journal_safety(analysis, chat_verdict=None):
    """Combine a journal's safety level with a chat verdict, taking the higher.

    Used when an entry arrives alongside a message. Escalating only - a calm
    journal never lowers a level the chat text raised, or vice versa.
    """
    journal_level = analysis.safety["level"]
    if chat_verdict is None:
        return journal_level
    return max((journal_level, chat_verdict["level"]),
               key=lambda level: SAFETY_ORDER[level])
