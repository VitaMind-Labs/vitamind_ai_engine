"""Request/response contracts for the Lumina HTTP service.

These Pydantic models are the single source of truth for the agent boundary. The
backend assembles an authorized context bundle; Lumina returns a structured
envelope; the backend validates and persists. Nothing here reads or writes a
database, and no field carries free-form instructions - only data.

Size caps are deliberate. An oversized bundle is rejected rather than silently
truncated, because a quietly trimmed memory list changes what the agent decides
without anyone noticing.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lumina import CONTRACT_VERSION
from lumina.taxonomy import LANGUAGES, TRACKS

Language = Literal["en", "ar"]
Track = Literal["ADHD", "BIPOLAR", "SCHIZOPHRENIA", "UNSPECIFIED"]
RequestClass = Literal["CHAT_SHORT", "CHAT_DEEP", "CHECKIN", "JOURNAL_ANALYSIS",
                       "SAFETY_CLASSIFICATION"]

MAX_TEXT = 10_000
MAX_JOURNAL_TEXT = 12_000
MAX_MEMORIES = 40
MAX_HISTORY_DAYS = 60
MAX_OUTCOMES = 200


class Strict(BaseModel):
    """Reject unknown fields: a typo in a caller must fail loudly, not be ignored."""
    model_config = ConfigDict(extra="forbid")


class CheckinPayload(Strict):
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    energy: float | None = Field(default=None, ge=0, le=10)
    stress: float | None = Field(default=None, ge=0, le=10)
    mood: float | None = Field(default=None, ge=0, le=10)
    focus: float | None = Field(default=None, ge=0, le=10)
    routine_stability: float | None = Field(default=None, ge=0, le=10)
    social_connection: float | None = Field(default=None, ge=0, le=10)
    task_completion: float | None = Field(default=None, ge=0, le=10)


class MemoryItem(Strict):
    category: str
    content: str = Field(max_length=500)
    source: str
    key: str = ""
    confidence: float = Field(default=1.0, ge=0, le=1)
    justification: str = Field(default="", max_length=300)


class OutcomeItem(Strict):
    intervention_id: str = Field(max_length=120)
    result: Literal["EFFECTIVE", "PARTIALLY_EFFECTIVE", "INEFFECTIVE", "UNCERTAIN"]
    engagement: Literal["offered", "accepted", "started", "completed",
                        "declined", "unknown"] = "unknown"
    at: str = ""


class JournalContext(Strict):
    """What the patient's latest analysed journal entry said, as structure only.

    Never the entry text (spec s46): the tier the journal assigned, the follow-ups it
    recommended, its cue categories, and timing. `follow_up_done` is the backend's
    record that a safety check-in already happened after that entry.
    """
    tier: str = Field(default="none", max_length=32)
    follow_ups: list[str] = Field(default_factory=list, max_length=10)
    cues: list[str] = Field(default_factory=list, max_length=10)
    hours_ago: float = Field(default=0, ge=0, le=24 * 30)
    follow_up_done: bool = False


class SafetyConfig(Strict):
    """Emergency resources come from the backend. The agent never invents one."""
    emergency_resources: list[str] = Field(default_factory=list, max_length=10)
    approved_messages_version: str = ""


class ContextBundle(Strict):
    """Everything Lumina is allowed to see for one turn."""
    request_id: str = Field(max_length=120)
    patient_id: str = Field(max_length=120, description="Opaque id. Never an email.")
    language: Language = "en"
    request_class: RequestClass = "CHAT_SHORT"
    track: Track = "UNSPECIFIED"
    secondary_track: Track | None = None

    text: str | None = Field(default=None, max_length=MAX_TEXT)
    checkin: CheckinPayload | None = None
    # Oldest first. Each entry is one day's reported signals.
    history: list[CheckinPayload] = Field(default_factory=list,
                                          max_length=MAX_HISTORY_DAYS)
    memory: list[MemoryItem] = Field(default_factory=list, max_length=MAX_MEMORIES)
    intervention_outcomes: list[OutcomeItem] = Field(default_factory=list,
                                                     max_length=MAX_OUTCOMES)
    journal_signals: dict[str, float] | None = None
    journal_context: JournalContext | None = None
    safety_config: SafetyConfig = Field(default_factory=SafetyConfig)
    contract_version: str = CONTRACT_VERSION


class JournalAnalyzeRequest(Strict):
    request_id: str = Field(max_length=120)
    patient_id: str = Field(max_length=120)
    entry_id: str = Field(max_length=120)
    text: str = Field(min_length=1, max_length=MAX_JOURNAL_TEXT)
    language: Language | None = None
    content_version: int = Field(default=1, ge=1)
    is_private: bool = False
    analysis_consent: bool = True
    contract_version: str = CONTRACT_VERSION


class StateRecomputeRequest(Strict):
    request_id: str = Field(max_length=120)
    patient_id: str = Field(max_length=120)
    track: Track = "UNSPECIFIED"
    history: list[CheckinPayload] = Field(default_factory=list,
                                          max_length=MAX_HISTORY_DAYS)
    contract_version: str = CONTRACT_VERSION


class OutcomeRequest(Strict):
    request_id: str = Field(max_length=120)
    patient_id: str = Field(max_length=120)
    outcomes: list[OutcomeItem] = Field(min_length=1, max_length=MAX_OUTCOMES)
    contract_version: str = CONTRACT_VERSION


__all__ = ["ContextBundle", "CheckinPayload", "JournalContext", "MemoryItem", "OutcomeItem",
           "SafetyConfig", "JournalAnalyzeRequest", "StateRecomputeRequest",
           "OutcomeRequest", "LANGUAGES", "TRACKS", "CONTRACT_VERSION"]
