"""Intervention catalog, eligibility filtering, ranking and outcome learning.

The model never invents an exercise. It chooses from an approved catalog, through
a filter chain that can only narrow (spec s11, s47):

    candidates -> track filter -> safety filter -> capacity filter
               -> contraindication filter -> per-patient suspension filter
               -> personalization ranking

Ranking is where personalization lives, and it is per-patient memory rather than
model weights: what worked for this person changes what they are offered next,
immediately, without retraining anything (s100).

Outcome learning is deliberately slow to conclude. One bad session does not
suspend an intervention; a repeated pattern does (s49).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..taxonomy import CAPACITY, SAFETY_ORDER

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"

OUTCOMES = ("EFFECTIVE", "PARTIALLY_EFFECTIVE", "INEFFECTIVE", "UNCERTAIN")
ENGAGEMENT = ("offered", "accepted", "started", "completed", "declined", "unknown")

# Capacity ordered least to most able; an intervention needs at least its own
# `min_capacity` to be offered.
#
# UNKNOWN sits with REDUCED rather than with VERY_LOW. Not knowing someone's
# capacity is not evidence that it is at its floor, and ranking it lowest made
# almost every intervention ineligible in plain chat - where no check-in has been
# filed yet, which is most of the time. Safety filtering is separate and
# unaffected by this; this only governs interaction load.
CAPACITY_RANK = {"VERY_LOW": 0, "REDUCED": 1, "UNKNOWN": 1, "NORMAL": 2, "HIGH": 3}

# How many comparable negative outcomes before an intervention is suspended for
# this patient. Two is deliberately not one: a single bad day is not evidence.
SUSPENSION_THRESHOLD = 2
MIN_OUTCOMES_FOR_RATE = 3


@dataclass
class Outcome:
    intervention_id: str
    result: str
    engagement: str = "unknown"
    pre_state: float | None = None
    post_state: float | None = None
    feedback: str = ""
    at: str = ""

    def __post_init__(self):
        if self.result not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.result!r}")
        if self.engagement not in ENGAGEMENT:
            raise ValueError(f"unknown engagement {self.engagement!r}")

    @property
    def negative(self):
        return self.result == "INEFFECTIVE" or self.engagement == "declined"

    def to_dict(self):
        return asdict(self)


@dataclass
class Candidate:
    intervention: dict
    score: float = 0.0
    reasons: list = field(default_factory=list)

    @property
    def id(self):
        return self.intervention["id"]

    def to_dict(self):
        return {"intervention_id": self.id, "goal": self.intervention["goal"],
                "score": round(self.score, 3), "reasons": self.reasons}


class InterventionCatalog:
    def __init__(self, path=None, allow_unreviewed=False):
        self.path = Path(path or CATALOG_PATH)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.version = raw["catalog_version"]
        self.never_claim = tuple(raw.get("never_claim", ()))
        self.allow_unreviewed = allow_unreviewed
        self.interventions = raw["interventions"]

    @property
    def unreviewed(self):
        return [i["id"] for i in self.interventions
                if i.get("approval_status") != "APPROVED"]

    def assert_production_ready(self):
        """Refuse to run on unreviewed clinical content unless explicitly allowed."""
        if self.unreviewed and not self.allow_unreviewed:
            raise RuntimeError(
                f"{len(self.unreviewed)} intervention(s) are not clinician-approved: "
                f"{', '.join(self.unreviewed[:5])}. Set allow_unreviewed=True for "
                "development, never for production.")

    def eligible(self, track, safety_level="NORMAL", capacity="NORMAL",
                 goal=None, suspended=(), contraindications=()):
        """Apply every filter in order, recording why each survivor qualified."""
        if capacity not in CAPACITY:
            raise ValueError(f"unknown capacity {capacity!r}")
        contraindications = {c.lower() for c in contraindications}
        suspended = set(suspended)
        survivors = []

        for intervention in self.interventions:
            if not intervention.get("active", False):
                continue
            if not self.allow_unreviewed and \
                    intervention.get("approval_status") != "APPROVED":
                continue
            if track not in intervention["track"]:
                continue
            if safety_level in intervention.get("excluded_safety_levels", []):
                continue
            # A CRISIS level excludes anything not explicitly cleared for it.
            if SAFETY_ORDER.get(safety_level, 0) >= SAFETY_ORDER["CRISIS"] and \
                    intervention.get("excluded_safety_levels"):
                continue
            if CAPACITY_RANK.get(capacity, 0) < \
                    CAPACITY_RANK.get(intervention.get("min_capacity", "NORMAL"), 2):
                continue
            if intervention["id"] in suspended:
                continue
            if any(c.lower() in contraindications
                   for c in intervention.get("contraindications", [])):
                continue
            if goal and intervention["goal"] != goal:
                continue
            survivors.append(Candidate(intervention, reasons=["eligible"]))
        return survivors


class OutcomeHistory:
    """Per-patient intervention outcomes and what they imply."""

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])

    def record(self, outcome):
        self.outcomes.append(outcome)
        return outcome

    def for_intervention(self, intervention_id):
        return [o for o in self.outcomes if o.intervention_id == intervention_id]

    def success_rate(self, intervention_id):
        """Only reported once there is enough evidence to mean anything (s12)."""
        history = self.for_intervention(intervention_id)
        if len(history) < MIN_OUTCOMES_FOR_RATE:
            return None
        good = sum(1 for o in history
                   if o.result in ("EFFECTIVE", "PARTIALLY_EFFECTIVE"))
        return round(good / len(history), 3)

    def suspended(self):
        """Interventions with repeated comparable negative outcomes."""
        counts = {}
        for outcome in self.outcomes:
            if outcome.negative:
                counts[outcome.intervention_id] = counts.get(outcome.intervention_id, 0) + 1
        return {i for i, n in counts.items() if n >= SUSPENSION_THRESHOLD}

    def to_memory_updates(self):
        """Express suspensions as INTERVENTION_RESPONSE memories, as the spec asks."""
        updates = []
        for intervention_id in sorted(self.suspended()):
            history = self.for_intervention(intervention_id)
            updates.append({
                "category": "INTERVENTION_RESPONSE",
                "key": intervention_id,
                "content": f"Repeated unhelpful or declined outcomes for "
                           f"{intervention_id}; suspended for this patient.",
                "source": "OBSERVED_PATTERN",
                "confidence": 0.9,
                "justification": f"{len(history)} recorded outcomes, "
                                 f"{sum(1 for o in history if o.negative)} negative",
            })
        return updates


def rank(candidates, history=None, preferred_goals=(), capacity="NORMAL",
         max_results=3):
    """Score eligible candidates by fit and by what has worked for this patient."""
    history = history or OutcomeHistory()
    for candidate in candidates:
        intervention = candidate.intervention
        score, reasons = 0.0, []

        if intervention["goal"] in preferred_goals:
            score += 3.0
            reasons.append("matches_current_goal")

        rate = history.success_rate(intervention["id"])
        if rate is not None:
            score += (rate - 0.5) * 4.0
            reasons.append(f"patient_success_rate={rate}")
        else:
            reasons.append("insufficient_outcome_history")

        # When capacity is low, prefer the shortest and easiest options.
        if CAPACITY_RANK.get(capacity, 2) <= 1:
            score += max(0.0, (6 - intervention.get("duration_minutes", 5)) * 0.4)
            if intervention.get("difficulty") == "LOW":
                score += 1.0
                reasons.append("low_difficulty_for_reduced_capacity")

        candidate.score = score
        candidate.reasons = reasons

    candidates.sort(key=lambda c: (-c.score, c.id))
    return candidates[:max_results]
