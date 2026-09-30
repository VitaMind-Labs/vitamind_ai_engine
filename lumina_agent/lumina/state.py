"""Patient state: a structured, non-diagnostic reading of how someone is doing.

The spec is blunt about the one rule that matters here (s9): the system must
distinguish observed from estimated from missing, and it must never fabricate a
measurement. So every dimension carries its own quality flag, a missing dimension
stays missing rather than being filled with a neutral-looking 5.0, and
`evidence_quality` summarises how much of the snapshot is actually grounded.

All dimensions live on one canonical 0-10 scale so that baselines, deviations and
track rules can be written once instead of per-unit. Raw sleep hours are kept
alongside the scaled value because hours are what a clinician reads and what the
patient actually reported.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field

from .taxonomy import OBSERVATION_QUALITY, STATE_DIMENSIONS

SCALE_MIN, SCALE_MAX = 0.0, 10.0

# Sleep is reported in hours; everything else arrives already on 0-10. The
# mapping is deliberately flat-topped: 7-9 hours is the plateau, and both ends
# fall away from it, because 12 hours is not "better" than 8.
SLEEP_HOURS_ANCHORS = ((0.0, 0.0), (3.0, 2.0), (5.0, 5.0), (6.5, 7.5),
                       (7.0, 9.0), (9.0, 9.0), (10.5, 7.0), (12.0, 5.0), (16.0, 2.0))

# Dimensions where a high number is a problem rather than a good sign. Change
# detection needs this to say "worse" instead of only "higher".
INVERTED = frozenset({"stress"})


def clamp(value, low=SCALE_MIN, high=SCALE_MAX):
    return max(low, min(high, float(value)))


def sleep_hours_to_scale(hours):
    """Piecewise-linear hours -> 0-10, saturating at both extremes."""
    hours = max(0.0, float(hours))
    anchors = SLEEP_HOURS_ANCHORS
    if hours <= anchors[0][0]:
        return anchors[0][1]
    if hours >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= hours <= x1:
            span = x1 - x0
            return y0 if span == 0 else y0 + (y1 - y0) * (hours - x0) / span
    return anchors[-1][1]


@dataclass
class Signal:
    """One dimension of the patient's state, with how we came to know it."""
    dimension: str
    value: float | None
    quality: str = "observed"
    source: str = "checkin"
    raw: float | None = None

    def __post_init__(self):
        if self.dimension not in STATE_DIMENSIONS:
            raise ValueError(f"unknown state dimension {self.dimension!r}")
        if self.quality not in OBSERVATION_QUALITY:
            raise ValueError(f"unknown observation quality {self.quality!r}")
        if self.value is None:
            # A value-less signal is only meaningful as an absence.
            if self.quality not in ("missing", "unknown"):
                raise ValueError("a signal without a value must be missing or unknown")
        else:
            self.value = clamp(self.value)

    def to_dict(self):
        return asdict(self)


@dataclass
class StateSnapshot:
    """The patient's state at one point in time. Descriptive, never diagnostic."""
    signals: dict = field(default_factory=dict)
    at: str = ""
    capacity: str = "UNKNOWN"
    distress: float | None = None

    def value(self, dimension):
        signal = self.signals.get(dimension)
        return signal.value if signal else None

    def quality(self, dimension):
        signal = self.signals.get(dimension)
        return signal.quality if signal else "missing"

    def observed(self):
        return {d: s for d, s in self.signals.items() if s.quality == "observed"}

    @property
    def evidence_quality(self):
        """Share of dimensions actually reported, not inferred or absent."""
        if not STATE_DIMENSIONS:
            return 0.0
        return round(len(self.observed()) / len(STATE_DIMENSIONS), 3)

    def to_dict(self):
        return {
            "at": self.at,
            "capacity": self.capacity,
            "distress": self.distress,
            "evidence_quality": self.evidence_quality,
            "signals": {d: s.to_dict() for d, s in self.signals.items()},
            "missing_dimensions": sorted(d for d in STATE_DIMENSIONS
                                         if self.quality(d) in ("missing", "unknown")),
            "is_diagnostic": False,
        }


def _finite(value):
    """A usable number, or None. NaN/inf, booleans and text are absent, not extreme:
    clamping NaN would otherwise turn garbage into a maximum reading."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_state(checkin=None, journal_signals=None, at=None):
    """Assemble a snapshot from a check-in and optional journal-derived signals.

    A check-in is first-person self-report and outranks anything inferred from
    journal text, so journal signals only fill dimensions the check-in left
    empty - and they are marked `estimated` when they do.
    """
    checkin = {k: _finite(v) for k, v in (checkin or {}).items()}
    journal_signals = {k: _finite(v) for k, v in (journal_signals or {}).items()}
    signals = {}

    sleep_hours = checkin.get("sleep_hours")
    if sleep_hours is not None:
        signals["sleep"] = Signal("sleep", sleep_hours_to_scale(sleep_hours),
                                  "observed", "checkin", raw=float(sleep_hours))

    for dimension in STATE_DIMENSIONS:
        if dimension in signals:
            continue
        if checkin.get(dimension) is not None:
            signals[dimension] = Signal(dimension, checkin[dimension], "observed", "checkin")
        elif journal_signals.get(dimension) is not None:
            signals[dimension] = Signal(dimension, journal_signals[dimension],
                                        "estimated", "journal_analysis")
        else:
            signals[dimension] = Signal(dimension, None, "missing", "none")

    snapshot = StateSnapshot(
        signals=signals,
        at=at or dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    snapshot.distress = _distress(snapshot)
    return snapshot


def _distress(snapshot):
    """A coarse roll-up of the dimensions that carry load, or None if unknown.

    Returning None when nothing was reported is the point: a confident 0.0 for a
    patient who answered nothing would be a fabricated measurement.
    """
    parts = []
    stress = snapshot.value("stress")
    if stress is not None:
        parts.append(stress)
    for dimension in ("mood", "energy", "social_connection"):
        value = snapshot.value(dimension)
        if value is not None:
            parts.append(SCALE_MAX - value)
    if not parts:
        return None
    return round(sum(parts) / len(parts) / SCALE_MAX, 3)
