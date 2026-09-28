"""Structured patient memory: typed, sourced, confidence-aware, versioned.

Memory is not conversation history. Lumina stores a small number of durable facts
about a patient - preferences, goals, what has worked, constraints a clinician
set - and retrieves only the ones relevant to the current turn (spec s40-s42).

Three rules shape the design:

* A correction from the patient outranks anything the system inferred (s41). When
  they contradict, the model's version is invalidated, not merged.
* A model inference never silently becomes a fact (s41). Below a confidence floor
  it is stored as a candidate that needs confirmation.
* Trivia is not stored. A memory has to be worth carrying into future turns, so
  the writer asks for a justification and short throwaway content is refused.

Memory is patient-specific state, not model weights (s100): writing one changes
this patient's experience immediately and retrains nothing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import asdict, dataclass, field

from .taxonomy import MEMORY_PRIORITY, MEMORY_TYPES
from .text import normalize

STATUSES = ("ACTIVE", "CANDIDATE", "SUPERSEDED", "INVALIDATED")

# Sources ordered by how much they are trusted, mirroring MEMORY_PRIORITY.
SOURCES = ("PATIENT_CORRECTION", "PATIENT_EXPLICIT_STATEMENT",
           "STRUCTURED_CHECK_IN", "OBSERVED_PATTERN", "MODEL_INFERENCE")

# An inference below this is a candidate, never an active fact.
CONFIDENCE_FLOOR = 0.75

# Categories a model may never assert on its own, however confident it is.
CLINICIAN_ONLY = frozenset({"CLINICIAN_CONSTRAINT"})

MIN_CONTENT_CHARS = 12

# Relevance weighting. Category priority dominates, then key match, then recency.
CATEGORY_WEIGHT = {
    "SAFETY": 5.0, "CLINICIAN_CONSTRAINT": 4.5, "PATIENT_CORRECTION": 4.0,
    "PREFERENCE": 3.0, "INTERVENTION_RESPONSE": 3.0, "GOAL": 2.5,
    "ORIENTATION": 2.0, "BASELINE": 1.5, "PATTERN": 1.5, "CONTEXT": 1.0,
}


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class Memory:
    category: str
    content: str
    source: str
    key: str = ""
    confidence: float = 1.0
    status: str = "ACTIVE"
    justification: str = ""
    created_at: str = field(default_factory=_now)
    last_confirmed_at: str = ""
    superseded_by: str | None = None
    id: str = ""

    def __post_init__(self):
        if self.category not in MEMORY_TYPES:
            raise ValueError(f"unknown memory category {self.category!r}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown memory source {self.source!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown memory status {self.status!r}")
        if not self.id:
            digest = hashlib.sha256(
                f"{self.category}:{self.key}:{normalize(self.content)}:{self.created_at}"
                .encode("utf-8")).hexdigest()
            self.id = "mem_" + digest[:16]
        if not self.key:
            self.key = self.category.lower()

    @property
    def priority(self):
        return MEMORY_PRIORITY.get(self.source, 0)

    @property
    def active(self):
        return self.status == "ACTIVE"

    def to_dict(self):
        return asdict(self)


class MemoryStore:
    """In-process memory for one patient. Persistence is the backend's job."""

    def __init__(self, memories=None):
        self.memories = list(memories or [])

    # -- writing ---------------------------------------------------------
    def propose(self, category, content, source, key="", confidence=1.0,
                justification=""):
        """Offer a memory. Returns (memory, action) where action explains the call.

        Actions: `created`, `candidate`, `refused`, `confirmed`, `superseded`.
        """
        content = (content or "").strip()
        if len(content) < MIN_CONTENT_CHARS:
            return None, "refused:too_trivial_to_carry_forward"
        if not justification:
            return None, "refused:no_justification"
        if category in CLINICIAN_ONLY and source != "PATIENT_CORRECTION" \
                and source not in ("STRUCTURED_CHECK_IN",):
            if source == "MODEL_INFERENCE":
                return None, "refused:clinician_constraint_cannot_be_inferred"

        key = key or category.lower()
        existing = self.find(category, key)

        # An identical restatement confirms rather than duplicates.
        for memory in existing:
            if memory.active and normalize(memory.content) == normalize(content):
                memory.last_confirmed_at = _now()
                memory.confidence = max(memory.confidence, confidence)
                return memory, "confirmed"

        status = "ACTIVE"
        if source == "MODEL_INFERENCE" and confidence < CONFIDENCE_FLOOR:
            status = "CANDIDATE"

        memory = Memory(category=category, content=content, source=source, key=key,
                        confidence=confidence, status=status,
                        justification=justification,
                        last_confirmed_at=_now() if status == "ACTIVE" else "")

        # A new statement on the same key replaces weaker-sourced ones.
        if status == "ACTIVE":
            for previous in existing:
                if not previous.active:
                    continue
                if memory.priority >= previous.priority:
                    previous.status = ("INVALIDATED"
                                       if source == "PATIENT_CORRECTION"
                                       else "SUPERSEDED")
                    previous.superseded_by = memory.id
                else:
                    # A weaker source cannot overwrite a stronger one; it is kept
                    # as a candidate so the disagreement stays visible.
                    memory.status = "CANDIDATE"

        self.memories.append(memory)
        return memory, ("candidate" if memory.status == "CANDIDATE"
                        else "superseded" if any(m.superseded_by == memory.id
                                                 for m in self.memories)
                        else "created")

    def correct(self, category, content, key="", justification="patient correction"):
        """Record a patient correction: the strongest source there is."""
        return self.propose(category, content, "PATIENT_CORRECTION", key=key,
                            confidence=1.0, justification=justification)

    def invalidate(self, memory_id, reason="invalidated"):
        for memory in self.memories:
            if memory.id == memory_id:
                memory.status = "INVALIDATED"
                memory.justification = f"{memory.justification} | {reason}".strip(" |")
                return memory
        return None

    # -- reading ---------------------------------------------------------
    def find(self, category=None, key=None):
        return [m for m in self.memories
                if (category is None or m.category == category)
                and (key is None or m.key == key)]

    def active(self):
        return [m for m in self.memories if m.active]

    def retrieve(self, keys=(), categories=(), limit=8):
        """Return the most relevant ACTIVE memories, highest first.

        Bounded on purpose: the spec forbids loading a patient's whole history
        into every interaction (s42, s71). Candidates are excluded - an
        unconfirmed inference must not steer a response.
        """
        keys = {k.lower() for k in keys}
        categories = set(categories)
        scored = []
        for memory in self.active():
            score = CATEGORY_WEIGHT.get(memory.category, 1.0)
            if memory.key.lower() in keys:
                score += 3.0
            if memory.category in categories:
                score += 2.0
            score += memory.priority * 0.5
            score += memory.confidence
            scored.append((score, memory))
        scored.sort(key=lambda pair: (-pair[0], pair[1].created_at))
        return [memory for _, memory in scored[:limit]]

    def to_dict(self):
        return {"count": len(self.memories),
                "active": len(self.active()),
                "memories": [m.to_dict() for m in self.memories]}
