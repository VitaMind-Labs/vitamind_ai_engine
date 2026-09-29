"""The decision engine: what Lumina is allowed to do on this turn.

This is the heart of the system and it is entirely rule-based on purpose. The
rules decide *what* happens - safety handling, whether an intervention is offered
and which one, what the response strategy is. The response layer only decides
*how* that is said (spec s13, s50, s51).

Priority is fixed and non-negotiable (s52):

    CRISIS > HIGH SAFETY > TRACK SAFETY > STATE > INTERVENTION
           > GENERAL SUPPORT > SMALL TALK

Nothing downstream may reorder it, and no generated text may reach a patient
without having passed through here first.

The output carries reason codes, never reasoning prose (s112): a decision is
auditable by replaying the codes, and no chain-of-thought is stored.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .interventions import rank
from .taxonomy import RESPONSE_STRATEGIES, SAFETY_ORDER

RULE_VERSION = "decision-rules-v2"

# Concern code -> (strategy, goal to retrieve an intervention for).
CONCERN_PLAN = {
    "TASK_INITIATION": ("MICRO_ACTION", "task_initiation"),
    "OVERLOAD": ("TASK_BREAKDOWN", "prioritization"),
    "ATTENTION": ("MICRO_ACTION", "distraction_reduction"),
    "ROUTINE_INSTABILITY": ("ROUTINE_SUPPORT", "routine_stabilization"),
    "FOCUS_CHANGE_FROM_BASELINE": ("STATE_MONITORING", None),
    "SLEEP_ENERGY_DIVERGENCE": ("STATE_MONITORING", "routine_stabilization"),
    "SLEEP_REDUCTION": ("SLEEP_SUPPORT", "sleep_consistency"),
    "ENERGY_ELEVATION": ("STATE_MONITORING", "activity_pacing"),
    "ROUTINE_DESTABILIZATION": ("ROUTINE_SUPPORT", "routine_stabilization"),
    "LOW_SLEEP": ("SLEEP_SUPPORT", "sleep_consistency"),
    "DISTRESS": ("GROUNDING", "stress_reduction"),
    "SOCIAL_WITHDRAWAL": ("ENCOURAGE_SUPPORT_CONNECTION", "social_connection"),
    "ROUTINE_SUPPORT_NEEDED": ("ROUTINE_SUPPORT", "routine_support"),
    "BASELINE_DEVIATION": ("STATE_MONITORING", None),
}


# What the patient asked for, when the state data shows no concern of its own.
# MEDICATION_MENTION and CLINICIAN_MENTION acknowledge and never advise: Lumina
# does not comment on prescriptions or treatment plans (spec s80, s116).
INTENT_PLAN = {
    "TASK_SUPPORT": ("MICRO_ACTION", "task_initiation"),
    "FOCUS": ("MICRO_ACTION", "distraction_reduction"),
    "SLEEP": ("SLEEP_SUPPORT", "sleep_consistency"),
    "STRESS": ("GROUNDING", "stress_reduction"),
    "ROUTINE": ("ROUTINE_SUPPORT", "routine_stabilization"),
    "SOCIAL": ("ENCOURAGE_SUPPORT_CONNECTION", "social_connection"),
    "ENERGY": ("STATE_MONITORING", None),
    "EMOTIONAL_SUPPORT": ("REFLECT", None),
    "JOURNAL": ("ACKNOWLEDGE", None),
    "CHECK_IN": ("ACKNOWLEDGE", None),
    "PROGRESS": ("FOLLOW_UP", None),
    "GOAL": ("FOLLOW_UP", None),
    "EXERCISE": ("ACKNOWLEDGE", None),
    "MEDICATION_MENTION": ("ACKNOWLEDGE", None),
    "CLINICIAN_MENTION": ("ACKNOWLEDGE", None),
    "QUESTION": ("CLARIFY", None),
    "GENERAL_CONVERSATION": ("ACKNOWLEDGE", None),
}


@dataclass
class Decision:
    type: str
    strategy: str
    reason_codes: list = field(default_factory=list)
    intervention_id: str | None = None
    intervention: dict | None = None
    alternatives: list = field(default_factory=list)
    safety_level: str = "NORMAL"
    capacity: str = "UNKNOWN"
    track: str = "UNSPECIFIED"
    review_flag: dict | None = None
    prohibited: tuple = ()
    requires_human_review: bool = False
    rule_version: str = RULE_VERSION

    def __post_init__(self):
        if self.strategy not in RESPONSE_STRATEGIES:
            raise ValueError(f"unknown response strategy {self.strategy!r}")

    def to_dict(self):
        data = asdict(self)
        data["prohibited"] = list(self.prohibited)
        # Only structured metadata is ever exposed - no reasoning text.
        data["stores_chain_of_thought"] = False
        return data


# Acts that close a conversational loop. Answering "thanks" with a sleep
# exercise reads as not listening, so below the safety tiers they are simply
# acknowledged - the concern was already raised when the check-in was answered.
CLOSING_ACTS = ("THANKS", "CLOSING")


def decide(*, safety, state, changes, capacity, track_readings, catalog,
           history=None, memories=(), goal_hint=None, intent=None, act=None):
    """Produce the single decision this turn is allowed to act on.

    `safety` is the fused verdict from lumina.safety, `capacity` a CapacityReading,
    `track_readings` the output of tracks.read_tracks. `act` is the conversational
    move (intent.conversation_act); it adds an ACT_* reason code the response
    layer uses for wording and never changes the safety tiers.
    """
    decision = _decide(safety=safety, state=state, changes=changes, capacity=capacity,
                       track_readings=track_readings, catalog=catalog, history=history,
                       memories=memories, goal_hint=goal_hint, intent=intent, act=act)
    if act and decision.type not in ("CRISIS_WORKFLOW", "ELEVATED_SAFETY_WORKFLOW",
                                     "SAFETY_CLARIFICATION"):
        decision.reason_codes.append(f"ACT_{act}")
    if intent and decision.type not in ("CRISIS_WORKFLOW",) and             f"INTENT_{intent}" not in decision.reason_codes:
        decision.reason_codes.append(f"TOPIC_{intent}")
    return decision


def _decide(*, safety, state, changes, capacity, track_readings, catalog,
            history=None, memories=(), goal_hint=None, intent=None, act=None):
    level = safety["level"]
    primary = track_readings[0]
    prohibited = tuple(dict.fromkeys(
        p for reading in track_readings for p in reading.prohibited))
    review_flag = next((r.review_flag for r in track_readings if r.review_flag), None)

    # 1. CRISIS - stop everything else.
    if SAFETY_ORDER[level] >= SAFETY_ORDER["CRISIS"]:
        return Decision(
            type="CRISIS_WORKFLOW", strategy="SAFETY_CHECK",
            reason_codes=["SAFETY_CRISIS", f"DECIDED_BY_{safety['decided_by'].upper()}"],
            safety_level=level, capacity="VERY_LOW", track=primary.track,
            prohibited=prohibited + ("intervention_offer", "small_talk"),
            requires_human_review=True, review_flag=review_flag)

    # 2. Elevated safety - support first, everything else deferred.
    if SAFETY_ORDER[level] >= SAFETY_ORDER["HIGH"]:
        candidates = catalog.eligible(
            primary.track, safety_level=level, capacity=capacity.level,
            goal="stress_reduction", suspended=(history.suspended() if history else ()))
        ranked = rank(candidates, history, ("stress_reduction",), capacity.level, 1)
        chosen = ranked[0] if ranked else None
        return Decision(
            type="ELEVATED_SAFETY_WORKFLOW", strategy="SAFETY_CHECK",
            reason_codes=["SAFETY_HIGH", f"CAPACITY_{capacity.level}"],
            intervention_id=chosen.id if chosen else None,
            intervention=chosen.intervention if chosen else None,
            safety_level=level, capacity=capacity.level, track=primary.track,
            prohibited=prohibited, requires_human_review=True, review_flag=review_flag)

    # 3. Uncertain safety is handled conservatively, never as normal.
    if level == "UNKNOWN":
        return Decision(
            type="SAFETY_CLARIFICATION", strategy="SAFETY_CHECK",
            reason_codes=["SAFETY_UNKNOWN", "CONSERVATIVE_DEFAULT"],
            safety_level=level, capacity=capacity.level, track=primary.track,
            prohibited=prohibited, requires_human_review=True, review_flag=review_flag)

    # 3b. "Thanks" / "bye" below every safety tier: close the loop warmly.
    if act in CLOSING_ACTS and intent in ("GENERAL_CONVERSATION", None, "UNKNOWN"):
        return Decision(
            type="GENERAL_SUPPORT", strategy="ACKNOWLEDGE",
            reason_codes=["CONVERSATION_CLOSE", f"CAPACITY_{capacity.level}"],
            safety_level=level, capacity=capacity.level, track=primary.track,
            prohibited=prohibited, review_flag=review_flag)

    # 4. Track concerns, most urgent first across every active track.
    concerns = sorted((c for reading in track_readings for c in reading.concerns),
                      key=lambda c: -c.priority)
    if concerns:
        top = concerns[0]
        strategy, goal = CONCERN_PLAN.get(top.code, ("ACKNOWLEDGE", None))
        goal = goal_hint or goal
        reason_codes = [f"CONCERN_{top.code}", f"CAPACITY_{capacity.level}",
                        f"TRACK_{primary.track}"]

        chosen, alternatives = None, []
        if goal:
            candidates = catalog.eligible(
                primary.track, safety_level=level, capacity=capacity.level, goal=goal,
                suspended=(history.suspended() if history else ()))
            if not candidates:
                # Nothing approved fits; say so rather than inventing an exercise.
                candidates = catalog.eligible(
                    primary.track, safety_level=level, capacity=capacity.level,
                    suspended=(history.suspended() if history else ()))
                if candidates:
                    reason_codes.append("GOAL_UNAVAILABLE_FELL_BACK_TO_TRACK")
            ranked = rank(candidates, history, (goal,), capacity.level, 3)
            if ranked:
                chosen = ranked[0]
                alternatives = [c.to_dict() for c in ranked[1:]]
            else:
                reason_codes.append("NO_ELIGIBLE_INTERVENTION")

        for memory in memories:
            if memory.category == "PREFERENCE":
                reason_codes.append("APPLIED_PATIENT_PREFERENCE")
                break

        return Decision(
            type="SUPPORT", strategy=strategy, reason_codes=reason_codes,
            intervention_id=chosen.id if chosen else None,
            intervention=chosen.intervention if chosen else None,
            alternatives=alternatives, safety_level=level, capacity=capacity.level,
            track=primary.track, prohibited=prohibited, review_flag=review_flag,
            requires_human_review=bool(review_flag))

    # 5. No state-derived concern, but the patient asked for something. Route on
    #    what they said rather than falling back to a generic clarification.
    plan = INTENT_PLAN.get(intent) if intent else None
    if plan:
        strategy, goal = plan
        reason_codes = [f"INTENT_{intent}", f"CAPACITY_{capacity.level}",
                        f"TRACK_{primary.track}", "NO_STATE_CONCERN"]
        chosen = None
        if goal:
            candidates = catalog.eligible(
                primary.track, safety_level=level, capacity=capacity.level, goal=goal,
                suspended=(history.suspended() if history else ()))
            ranked = rank(candidates, history, (goal,), capacity.level, 1)
            chosen = ranked[0] if ranked else None
            if not chosen:
                reason_codes.append("NO_ELIGIBLE_INTERVENTION")
        return Decision(
            type="SUPPORT", strategy=strategy, reason_codes=reason_codes,
            intervention_id=chosen.id if chosen else None,
            intervention=chosen.intervention if chosen else None,
            safety_level=level, capacity=capacity.level, track=primary.track,
            prohibited=prohibited, review_flag=review_flag,
            requires_human_review=bool(review_flag))

    # 6. Free text we could not route: listen and invite more, rather than
    #    commenting on a day the patient did not describe.
    if intent == "UNKNOWN":
        return Decision(
            type="GENERAL_SUPPORT", strategy="CLARIFY",
            reason_codes=["OPEN_LISTENING", f"CAPACITY_{capacity.level}"],
            safety_level=level, capacity=capacity.level, track=primary.track,
            prohibited=prohibited, review_flag=review_flag)

    # 7. Nothing pressing, but too little evidence to say anything about state.
    if state.evidence_quality == 0.0:
        return Decision(
            type="GENERAL_SUPPORT", strategy="CLARIFY",
            reason_codes=["NO_REPORTED_SIGNALS", "INSUFFICIENT_EVIDENCE"],
            safety_level=level, capacity=capacity.level, track=primary.track,
            prohibited=prohibited, review_flag=review_flag)

    # 8. Steady day.
    return Decision(
        type="GENERAL_SUPPORT", strategy="ACKNOWLEDGE",
        reason_codes=["NO_ACTIVE_CONCERN", f"CAPACITY_{capacity.level}"],
        safety_level=level, capacity=capacity.level, track=primary.track,
        prohibited=prohibited, review_flag=review_flag)
