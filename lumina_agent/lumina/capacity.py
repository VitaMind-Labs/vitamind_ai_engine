"""Capacity estimation: how much interaction the patient can take right now.

This is a UX decision, not a clinical one (spec s39). Capacity controls how long
a reply is, how many questions it asks, and how many choices it offers - nothing
else. It never appears in a report as a finding about the person.

Deliberately deterministic and cheap: it runs on every turn, and a rule someone
can read beats a model nobody can audit for a decision this small.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .taxonomy import CAPACITY

RULE_VERSION = "capacity-rules-v1"

# How the interaction is shaped at each level. These are hard caps applied to the
# generated response, not suggestions.
CONSTRAINTS = {
    "HIGH": {"max_sentences": 6, "max_questions": 2, "max_choices": 3,
             "max_actions": 2, "allow_long_exercises": True},
    "NORMAL": {"max_sentences": 5, "max_questions": 1, "max_choices": 3,
               "max_actions": 1, "allow_long_exercises": True},
    "REDUCED": {"max_sentences": 3, "max_questions": 1, "max_choices": 2,
                "max_actions": 1, "allow_long_exercises": False},
    "VERY_LOW": {"max_sentences": 2, "max_questions": 1, "max_choices": 0,
                 "max_actions": 1, "allow_long_exercises": False},
    # Not knowing is not a reason to overload someone.
    "UNKNOWN": {"max_sentences": 3, "max_questions": 1, "max_choices": 2,
                "max_actions": 1, "allow_long_exercises": False},
}

CONFIG = {
    "very_low_distress": 0.78,
    "reduced_distress": 0.62,
    "high_distress_ceiling": 0.30,
    "low_energy": 3.0,
    "high_stress": 7.5,
    "low_focus": 3.0,
    "high_energy": 7.0,
    "min_evidence_for_high": 0.5,
}


@dataclass
class CapacityReading:
    level: str
    reasons: tuple = ()
    evidence_quality: float = 0.0
    rule_version: str = RULE_VERSION
    is_diagnostic: bool = False

    def __post_init__(self):
        if self.level not in CAPACITY:
            raise ValueError(f"unknown capacity {self.level!r}")

    @property
    def constraints(self):
        return dict(CONSTRAINTS[self.level])

    def to_dict(self):
        return {**asdict(self), "constraints": self.constraints}


def estimate(state, safety_level="NORMAL", config=None):
    """Estimate capacity from the current snapshot and the fused safety level."""
    config = {**CONFIG, **(config or {})}
    reasons = []

    # Safety dominates: someone in crisis gets the shortest possible interaction.
    if safety_level == "CRISIS":
        return CapacityReading("VERY_LOW", ("safety_crisis",), state.evidence_quality)
    if safety_level == "HIGH":
        reasons.append("safety_high")

    energy = state.value("energy")
    stress = state.value("stress")
    focus = state.value("focus")
    distress = state.distress

    if state.evidence_quality == 0.0 and not reasons:
        return CapacityReading("UNKNOWN", ("no_reported_signals",), 0.0)

    if distress is not None and distress >= config["very_low_distress"]:
        reasons.append("very_high_distress")
    if distress is not None and distress >= config["reduced_distress"]:
        reasons.append("elevated_distress")
    if energy is not None and energy <= config["low_energy"]:
        reasons.append("low_energy")
    if stress is not None and stress >= config["high_stress"]:
        reasons.append("high_stress")
    if focus is not None and focus <= config["low_focus"]:
        reasons.append("low_focus")

    if "very_high_distress" in reasons or reasons.count("low_energy") + \
            reasons.count("high_stress") + reasons.count("low_focus") >= 3:
        level = "VERY_LOW"
    elif "safety_high" in reasons or "elevated_distress" in reasons or len(reasons) >= 2:
        level = "REDUCED"
    elif reasons:
        level = "REDUCED"
    elif (distress is not None and distress <= config["high_distress_ceiling"]
          and energy is not None and energy >= config["high_energy"]
          and state.evidence_quality >= config["min_evidence_for_high"]):
        level = "HIGH"
        reasons.append("low_distress_good_energy")
    else:
        level = "NORMAL"
        reasons.append("no_limiting_signals")

    return CapacityReading(level, tuple(reasons), state.evidence_quality)
