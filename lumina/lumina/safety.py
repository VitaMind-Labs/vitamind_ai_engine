"""Safety fusion. Deterministic rules decide; the learned head only escalates.

The spec is unambiguous about the ordering (s10, s52, s124): safety runs before
any response is produced, and generation must never overrule it. This module is
where that ordering is enforced, and it is built on two independent detectors so
that neither one failing can silently produce a NORMAL verdict:

* Rules - the complete journal_ai analyzer: its cue lexicon, its own
  small classifier, and its clause-scoped fusion policy. The scoping is the point:
  "my brother said he wanted to die" and "a few years ago I felt that way" are
  not treated as present first-person risk, and "this traffic is killing me" is
  recognised as an idiom. Only this detector can reach CRISIS.

  It must be driven through `JournalSentinel.analyze`, not by calling its `fuse`
  with a null model context - that null path is the journal's own
  model-unavailable branch, which deliberately flags *everything* for review and
  would rate a benign sentence HIGH.

* Learned head - the calibrated three-level distress signal trained in
  training/train_safety.py. It generalises to phrasing the lexicon never
  anticipated, which is exactly what a fixed regex list is bad at.

The fusion is deliberately one-directional: `final = max(rules, model)`. The
model can raise a level the lexicon missed. It can never lower one the lexicon
raised, and it cannot reach CRISIS on its own, because it was never trained on
enough distinct crisis meanings to be trusted with that call.

An unavailable model is not an absence of risk: if the head fails to load or
throws, the verdict falls back to the rules and the failure is reported in the
output rather than swallowed.
"""
from __future__ import annotations

from pathlib import Path

from .taxonomy import SAFETY_LEVELS, SAFETY_ORDER

# Journal tier -> Lumina safety level, following the journal's own *action*
# semantics rather than the ordering of its tier names:
#
#   none / low        -> no support panel, optional reflection  -> NORMAL
#   moderate          -> support panel, review recommended      -> ELEVATED
#   moderate_flagged  -> the same, plus "needs a second look"   -> ELEVATED
#   high              -> crisis support, urgent priority        -> CRISIS
#
# `moderate_flagged` is a *review* marker on a moderate signal - the journal
# returns it when its own model was unavailable or disagreed with the lexicon -
# not a level above moderate. Reading it as HIGH turned ordinary ADHD overload
# ("ten things to do and I'm doing none of them") into a safety workflow, which
# would make the product unusable for the track it is meant to serve.
TIER_TO_LEVEL = {"none": "NORMAL", "low": "NORMAL", "moderate": "ELEVATED",
                 "moderate_flagged": "ELEVATED", "high": "CRISIS"}

# Tiers that additionally mean "a human should look at this", independent of the
# level they map to.
TIERS_NEEDING_REVIEW = frozenset({"moderate_flagged", "high"})

# An abstaining head is uncertainty, not safety. Uncertainty is treated as
# ELEVATED so the conservative branch is taken (spec s124).
UNKNOWN_FLOOR = "ELEVATED"


def _max_level(*levels):
    return max(levels, key=lambda level: SAFETY_ORDER[level])


class SafetyEngine:
    """Fuses deterministic cues with the learned distress head."""

    def __init__(self, model=None, rules_enabled=True):
        self.model = model
        self.rules_enabled = rules_enabled
        self._sentinel = None
        self.rules_version = None
        if rules_enabled:
            from journal_ai.lexicon import VERSION as LEXICON_VERSION
            from journal_ai.pipeline import JournalSentinel
            # Loaded once; analyze() is pure and holds no per-patient state.
            self._sentinel = JournalSentinel()
            self.rules_version = LEXICON_VERSION

    @classmethod
    def load(cls, folder=None, rules_enabled=True):
        """Load the trained head; a missing model degrades to rules-only."""
        from .linear import MultiHeadLinear
        folder = Path(folder or Path(__file__).resolve().parents[1]
                      / "artifacts" / "models" / "safety")
        model = None
        try:
            model = MultiHeadLinear.load(folder)
        except (OSError, ValueError, KeyError):
            model = None
        return cls(model=model, rules_enabled=rules_enabled)

    # -- detectors --------------------------------------------------------
    def _rule_level(self, text):
        if not self.rules_enabled or self._sentinel is None:
            return None, {"available": False, "reason": "rules_disabled"}
        try:
            result = self._sentinel.analyze(text, include_audit=True)
        except Exception as error:  # a rules fault must not read as "safe"
            return None, {"available": False, "reason": type(error).__name__}
        tier = result["tier"]
        audit = result.get("audit", {})
        return TIER_TO_LEVEL.get(tier, UNKNOWN_FLOOR), {
            "available": True,
            "tier": tier,
            "reason": audit.get("fusion_reason"),
            "lexicon_version": self.rules_version,
            "journal_model_available": result["model"]["available"],
            "needs_review": tier in TIERS_NEEDING_REVIEW,
            "cues": sorted({h["category"] for h in audit.get("hits", [])}),
        }

    def _model_level(self, text):
        if self.model is None:
            return None, {"available": False, "reason": "model_unavailable"}
        try:
            prediction = self.model.predict(text)["safety_signal"]
        except Exception as error:  # never let a model fault mean "safe"
            return None, {"available": False, "reason": type(error).__name__}
        label = prediction["label"]
        level = UNKNOWN_FLOOR if label == "UNKNOWN" else label
        return level, {
            "available": True,
            "label": label,
            "predicted": prediction["predicted"],
            "confidence": prediction["confidence"],
            "abstained": prediction["abstain"],
            "model_version": self.model.version,
        }

    # -- fusion -----------------------------------------------------------
    def assess(self, text):
        """Return the fused safety verdict with both detectors' reasoning."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        rule_level, rule_detail = self._rule_level(text)
        model_level, model_detail = self._model_level(text)

        available = [level for level in (rule_level, model_level) if level is not None]
        if not available:
            # Both detectors are gone. Refusing to answer is the only honest
            # result; a caller must treat this as a degraded-service condition.
            return {"level": "UNKNOWN", "decided_by": "none",
                    "both_detectors_unavailable": True,
                    "rules": rule_detail, "model": model_detail,
                    "requires_human_review": True,
                    "clinically_validated": False}

        level = _max_level(*available)
        if rule_level is not None and SAFETY_ORDER[rule_level] >= SAFETY_ORDER.get(level, 0):
            decided_by = "rules"
        else:
            decided_by = "model_escalation"

        return {
            "level": level,
            "decided_by": decided_by,
            "rules": rule_detail,
            "model": model_detail,
            "model_can_reach_crisis": False,
            "escalated_by_model": bool(
                rule_level is not None and model_level is not None
                and SAFETY_ORDER[model_level] > SAFETY_ORDER[rule_level]),
            # Either a severe level, or a tier the rules explicitly flagged.
            "requires_human_review": bool(
                SAFETY_ORDER[level] >= SAFETY_ORDER["HIGH"]
                or rule_detail.get("needs_review")),
            "clinically_validated": False,
        }


def assert_never_lowers(rule_level, fused_level):
    """Invariant used by the tests: fusion may raise a level, never lower one."""
    if SAFETY_ORDER[fused_level] < SAFETY_ORDER[rule_level]:
        raise AssertionError(
            f"fusion lowered {rule_level} to {fused_level}; rules are authoritative")
    return True


__all__ = ["SafetyEngine", "SAFETY_LEVELS", "SAFETY_ORDER", "TIER_TO_LEVEL",
           "assert_never_lowers"]
