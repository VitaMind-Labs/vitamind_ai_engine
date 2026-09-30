"""The three support tracks, kept genuinely separate (spec s32-s35, s78-s79).

Each track reads the same state and change objects and says what *this* track
cares about: which concerns are active, what the response should prioritise, and
what language is forbidden. They do not share rules, so a change to the bipolar
thresholds cannot alter ADHD behaviour, and each can be evaluated alone.

Two hard limits apply to every track:

* A track never changes itself. The strongest thing any of them can emit is a
  `TRACK_REVIEW_FLAG` carrying its evidence, for a human to consider (s79).
* No track names an episode, a diagnosis, or a medication decision (s80). They
  describe reported patterns and deviations from the patient's own baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .taxonomy import TRACKS

RULE_VERSION = "track-rules-v1"


@dataclass
class Concern:
    code: str
    dimension: str = ""
    evidence: tuple = ()
    priority: int = 1

    def to_dict(self):
        return asdict(self)


@dataclass
class TrackReading:
    track: str
    concerns: list = field(default_factory=list)
    goals: tuple = ()
    review_flag: dict | None = None
    prohibited: tuple = ()
    rule_version: str = RULE_VERSION
    is_diagnostic: bool = False

    @property
    def top_concern(self):
        if not self.concerns:
            return None
        return sorted(self.concerns, key=lambda c: -c.priority)[0]

    def to_dict(self):
        return {"track": self.track,
                "concerns": [c.to_dict() for c in self.concerns],
                "goals": list(self.goals),
                "review_flag": self.review_flag,
                "prohibited": list(self.prohibited),
                "rule_version": self.rule_version,
                "is_diagnostic": False}


def _change(changes, dimension):
    for change in changes:
        if change.dimension == dimension:
            return change
    return None


class Track:
    name = "UNSPECIFIED"
    # Wording this track must never produce, enforced by the response guardrails.
    prohibited = ("diagnosis", "episode_naming", "medication_advice")
    goals = ()

    def read(self, state, changes, baseline):
        raise NotImplementedError

    def _base(self, concerns, review_flag=None):
        return TrackReading(track=self.name, concerns=concerns, goals=self.goals,
                            review_flag=review_flag, prohibited=self.prohibited)


class ADHDTrack(Track):
    """Execution support: starting, breaking down, prioritising, staying with it."""
    name = "ADHD"
    goals = ("task_initiation", "task_decomposition", "prioritization",
             "routine", "overwhelm_reduction")

    LOW_TASK_COMPLETION = 3.5
    LOW_FOCUS = 3.5
    HIGH_STRESS = 7.0
    LOW_ROUTINE = 4.0

    def read(self, state, changes, baseline):
        concerns = []
        task = state.value("task_completion")
        focus = state.value("focus")
        stress = state.value("stress")
        routine = state.value("routine_stability")

        if task is not None and task <= self.LOW_TASK_COMPLETION:
            concerns.append(Concern("TASK_INITIATION", "task_completion",
                                    (f"task_completion={task}",), priority=4))
        if focus is not None and focus <= self.LOW_FOCUS:
            concerns.append(Concern("ATTENTION", "focus", (f"focus={focus}",), priority=3))
        if stress is not None and stress >= self.HIGH_STRESS:
            # For ADHD, high stress alongside stalled tasks usually reads as
            # overload rather than anxiety, and the answer is less scope.
            concerns.append(Concern("OVERLOAD", "stress", (f"stress={stress}",), priority=4))
        if routine is not None and routine <= self.LOW_ROUTINE:
            concerns.append(Concern("ROUTINE_INSTABILITY", "routine_stability",
                                    (f"routine_stability={routine}",), priority=2))

        focus_change = _change(changes, "focus")
        if focus_change and focus_change.significance == "SIGNIFICANT_CHANGE":
            concerns.append(Concern("FOCUS_CHANGE_FROM_BASELINE", "focus",
                                    (f"delta={focus_change.delta}",), priority=3))
        return self._base(concerns)


class BipolarTrack(Track):
    """Longitudinal stability: sleep, energy, activity, routine - versus baseline.

    The sleep-down/energy-up combination is the one this track watches hardest,
    and it is also the one where overreach is most tempting. The output is
    `SLEEP_ENERGY_DIVERGENCE` and a monitoring posture - never an episode name.
    """
    name = "BIPOLAR"
    goals = ("routine_stabilization", "sleep_consistency", "activity_pacing",
             "stress_reduction", "daily_structure")
    prohibited = Track.prohibited + ("mania_claim", "episode_prediction")

    LOW_SLEEP = 4.0
    HIGH_ENERGY = 7.5

    def read(self, state, changes, baseline):
        concerns = []
        sleep_change = _change(changes, "sleep")
        energy_change = _change(changes, "energy")
        routine_change = _change(changes, "routine_stability")

        sleep_down = sleep_change and sleep_change.direction == "DECREASE"
        energy_up = energy_change and energy_change.direction == "INCREASE"

        if sleep_down and energy_up:
            evidence = (f"sleep {sleep_change.baseline_value}->{sleep_change.recent_value}",
                        f"energy {energy_change.baseline_value}->{energy_change.recent_value}",
                        f"persistence_days={min(sleep_change.persistence_days, energy_change.persistence_days)}")
            concerns.append(Concern("SLEEP_ENERGY_DIVERGENCE", "sleep", evidence, priority=5))
        elif sleep_down:
            concerns.append(Concern("SLEEP_REDUCTION", "sleep",
                                    (f"delta={sleep_change.delta}",), priority=4))
        elif energy_up:
            concerns.append(Concern("ENERGY_ELEVATION", "energy",
                                    (f"delta={energy_change.delta}",), priority=3))

        if routine_change and routine_change.direction == "DECREASE":
            concerns.append(Concern("ROUTINE_DESTABILIZATION", "routine_stability",
                                    (f"delta={routine_change.delta}",), priority=3))

        sleep = state.value("sleep")
        if sleep is not None and sleep <= self.LOW_SLEEP and not sleep_down:
            concerns.append(Concern("LOW_SLEEP", "sleep", (f"sleep={sleep}",), priority=3))

        # A sustained, significant divergence is worth a human look - and that is
        # the ceiling. The track does not reassign itself.
        review_flag = None
        divergence = [c for c in concerns if c.code == "SLEEP_ENERGY_DIVERGENCE"]
        if divergence and sleep_change.significance == "SIGNIFICANT_CHANGE":
            review_flag = {
                "type": "TRACK_REVIEW_FLAG",
                "reason": "sustained_sleep_energy_divergence_from_personal_baseline",
                "evidence": list(divergence[0].evidence),
                "action": "clinician_review_suggested",
                "not_a_diagnosis": True,
                "track_unchanged": True,
            }
        return self._base(concerns, review_flag)


class SchizophreniaTrack(Track):
    """Grounding, routine and daily function, with a strict language rule.

    Acknowledging distress is required. Confirming an unverified explanation for
    it is forbidden (s35, s58) - so is arguing against it. The response engine
    reads `ACKNOWLEDGE_WITHOUT_VALIDATING_INTERPRETATION` from here and applies
    calm, short, non-confrontational phrasing.
    """
    name = "SCHIZOPHRENIA"
    goals = ("grounding", "routine_support", "stress_management",
             "daily_living", "social_connection", "self_care")
    prohibited = Track.prohibited + ("validate_unverified_interpretation",
                                     "confront_or_dispute_belief",
                                     "dramatic_metaphor")

    LOW_SOCIAL = 3.5
    HIGH_STRESS = 7.0
    LOW_ROUTINE = 4.0

    def read(self, state, changes, baseline):
        concerns = []
        social = state.value("social_connection")
        stress = state.value("stress")
        routine = state.value("routine_stability")

        if stress is not None and stress >= self.HIGH_STRESS:
            concerns.append(Concern("DISTRESS", "stress", (f"stress={stress}",), priority=4))
        if social is not None and social <= self.LOW_SOCIAL:
            concerns.append(Concern("SOCIAL_WITHDRAWAL", "social_connection",
                                    (f"social_connection={social}",), priority=3))
        if routine is not None and routine <= self.LOW_ROUTINE:
            concerns.append(Concern("ROUTINE_SUPPORT_NEEDED", "routine_stability",
                                    (f"routine_stability={routine}",), priority=3))

        sleep_change = _change(changes, "sleep")
        if sleep_change and sleep_change.direction == "DECREASE":
            concerns.append(Concern("SLEEP_REDUCTION", "sleep",
                                    (f"delta={sleep_change.delta}",), priority=3))
        return self._base(concerns)


class UnspecifiedTrack(Track):
    """Used before orientation, or when Mira could not identify a clear track."""
    name = "UNSPECIFIED"
    goals = ("routine", "stress_reduction", "daily_structure")

    def read(self, state, changes, baseline):
        concerns = []
        stress = state.value("stress")
        if stress is not None and stress >= 7.0:
            concerns.append(Concern("DISTRESS", "stress", (f"stress={stress}",), priority=3))
        for change in changes:
            if change.significance == "SIGNIFICANT_CHANGE":
                concerns.append(Concern("BASELINE_DEVIATION", change.dimension,
                                        (f"delta={change.delta}",), priority=2))
        return self._base(concerns)


REGISTRY = {"ADHD": ADHDTrack, "BIPOLAR": BipolarTrack,
            "SCHIZOPHRENIA": SchizophreniaTrack, "UNSPECIFIED": UnspecifiedTrack}


def get_track(name):
    if name not in TRACKS:
        raise ValueError(f"unknown track {name!r}; tracks are {TRACKS}")
    return REGISTRY[name]()


def read_tracks(primary, state, changes, baseline, secondary=None):
    """Read the primary track, and a secondary one if the patient has both.

    Where they disagree the more conservative reading wins, which in practice
    means concerns from both are kept rather than the secondary being dropped.
    """
    readings = [get_track(primary).read(state, changes, baseline)]
    if secondary and secondary != primary:
        readings.append(get_track(secondary).read(state, changes, baseline))
    return readings
