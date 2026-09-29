"""Personal baseline and change detection. Deterministic, versioned, auditable.

Everything here compares a patient to *their own* recent history, never to a
population norm - which is the whole point for the bipolar track, where four
hours of sleep means something different for someone who normally sleeps five
than for someone who normally sleeps eight (spec s34, s36).

Two rules keep this honest:

* Below the minimum number of observations a dimension is `INSUFFICIENT_DATA`.
  It does not get a mean with two data points behind it.
* Every detected change records how it was computed - the dimension, direction,
  baseline value, recent value, how many days it persisted, and the rule version
  that decided. A change nobody can reconstruct is not evidence.

A detected change is an observation, never a diagnosis. `SIGNIFICANT_CHANGE` is
the strongest thing this module will ever say.
"""
from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import asdict, dataclass, field

from .state import INVERTED, SCALE_MAX
from .taxonomy import STATE_DIMENSIONS, TRENDS

RULE_VERSION = "baseline-rules-v1"

# Thresholds live here rather than inline so a change to them is a visible,
# reviewable edit with a version bump.
CONFIG = {
    "calibration_days": 14,
    "min_observations": 5,
    "min_observations_for_variance": 7,
    "recent_window": 3,
    "trend_window": 5,
    # A deviation counts when it clears both an absolute floor and the patient's
    # own spread. The floor stops a very stable patient from tripping an alert on
    # noise; the spread stops a volatile patient from never tripping one.
    "deviation_absolute": 1.5,
    "deviation_sigma": 1.25,
    "significant_absolute": 2.5,
    "significant_sigma": 2.0,
    "volatile_sigma": 2.2,
    "persistence_days_for_significant": 2,
}


@dataclass
class DimensionBaseline:
    dimension: str
    status: str = "INSUFFICIENT_DATA"
    mean: float | None = None
    stdev: float | None = None
    observations: int = 0
    window_days: int = 0

    @property
    def reliable(self):
        return self.status == "ESTABLISHED"

    def to_dict(self):
        return asdict(self)


@dataclass
class Change:
    """One detected deviation, carrying the arithmetic that produced it."""
    dimension: str
    direction: str
    comparison: str = "PERSONAL_BASELINE"
    baseline_value: float | None = None
    recent_value: float | None = None
    delta: float | None = None
    sigma: float | None = None
    persistence_days: int = 0
    certainty: str = "low"
    significance: str = "CHANGE"
    rule_version: str = RULE_VERSION
    is_diagnostic: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class BaselineProfile:
    dimensions: dict = field(default_factory=dict)
    calibrating: bool = True
    computed_at: str = ""
    rule_version: str = RULE_VERSION

    def get(self, dimension):
        return self.dimensions.get(dimension)

    def to_dict(self):
        return {"calibrating": self.calibrating, "computed_at": self.computed_at,
                "rule_version": self.rule_version,
                "dimensions": {d: b.to_dict() for d, b in self.dimensions.items()}}


def _series(history, dimension):
    """Ordered values for one dimension, skipping anything not actually reported."""
    values = []
    for snapshot in history:
        signal = snapshot.signals.get(dimension)
        if signal and signal.value is not None and signal.quality == "observed":
            values.append(signal.value)
    return values


def compute_baseline(history, config=None, at=None, exclude_recent=0):
    """Build a baseline from a list of StateSnapshots, oldest first.

    `exclude_recent` holds back the days that change detection is about to test.
    Without it the deviant days sit inside their own reference window, inflating
    the standard deviation and hiding the very shift we are looking for - a
    three-night sleep drop measured against a baseline containing those three
    nights reads as ordinary spread.
    """
    config = {**CONFIG, **(config or {})}
    reference = history[:-exclude_recent] if exclude_recent else history
    if len(reference) < config["min_observations"]:
        # Too little history to hold anything back; use what there is rather than
        # reporting INSUFFICIENT_DATA for a patient who has been checking in.
        reference = history
    window = reference[-config["calibration_days"]:]
    dimensions = {}
    for dimension in STATE_DIMENSIONS:
        values = _series(window, dimension)
        baseline = DimensionBaseline(dimension=dimension, observations=len(values),
                                     window_days=len(window))
        if len(values) >= config["min_observations"]:
            baseline.status = "ESTABLISHED"
            baseline.mean = round(statistics.fmean(values), 3)
            if len(values) >= config["min_observations_for_variance"]:
                baseline.stdev = round(statistics.stdev(values), 3)
        dimensions[dimension] = baseline

    established = [b for b in dimensions.values() if b.reliable]
    return BaselineProfile(
        dimensions=dimensions,
        calibrating=len(established) == 0,
        computed_at=at or dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def _direction(dimension, delta):
    """Say whether a move is an increase or a decrease, and whether that is worse."""
    rising = delta > 0
    worse = rising if dimension in INVERTED else not rising
    return ("INCREASE" if rising else "DECREASE"), worse


def detect_changes(history, baseline, config=None):
    """Compare the recent window against the baseline for every dimension."""
    config = {**CONFIG, **(config or {})}
    recent_window = history[-config["recent_window"]:]
    changes = []

    for dimension in STATE_DIMENSIONS:
        dimension_baseline = baseline.get(dimension)
        if dimension_baseline is None or not dimension_baseline.reliable:
            continue
        recent_values = _series(recent_window, dimension)
        if not recent_values:
            continue

        recent = statistics.fmean(recent_values)
        delta = recent - dimension_baseline.mean
        spread = dimension_baseline.stdev
        # A spread of zero is not a missing estimate - it means this patient has
        # been perfectly steady, which makes any movement *more* meaningful, not
        # less. Dividing by it is undefined, so sigma is left unset and the
        # absolute floor decides alone rather than the check being skipped.
        has_spread = spread is not None and spread > 0
        sigma = (abs(delta) / spread) if has_spread else None

        clears_absolute = abs(delta) >= config["deviation_absolute"]
        clears_sigma = sigma is not None and sigma >= config["deviation_sigma"]
        if not (clears_absolute and (clears_sigma or not has_spread)):
            continue

        persistence = 0
        for snapshot in reversed(history):
            value = snapshot.value(dimension)
            if value is None or snapshot.quality(dimension) != "observed":
                break
            if abs(value - dimension_baseline.mean) >= config["deviation_absolute"]:
                persistence += 1
            else:
                break

        direction, worse = _direction(dimension, delta)
        significant = (abs(delta) >= config["significant_absolute"]
                       and (not has_spread or sigma >= config["significant_sigma"])
                       and persistence >= config["persistence_days_for_significant"])

        if not has_spread:
            certainty = "low"
        elif significant:
            certainty = "moderate"
        else:
            certainty = "low" if persistence < 2 else "moderate"

        changes.append(Change(
            dimension=dimension,
            direction=direction,
            baseline_value=dimension_baseline.mean,
            recent_value=round(recent, 3),
            delta=round(delta, 3),
            sigma=round(sigma, 3) if sigma is not None else None,
            persistence_days=persistence,
            certainty=certainty,
            significance="SIGNIFICANT_CHANGE" if significant else "CHANGE",
        ))
        changes[-1].worsening = worse
    return changes


def trend(history, dimension, config=None):
    """Describe the recent direction of one dimension in plain, non-clinical terms."""
    config = {**CONFIG, **(config or {})}
    values = _series(history[-config["trend_window"]:], dimension)
    if len(values) < 3:
        return TRENDS[TRENDS.index("INSUFFICIENT_DATA")]

    spread = statistics.stdev(values) if len(values) >= 2 else 0.0
    # Average step direction, which is robust enough for a 5-point window and
    # does not pretend to be a fitted regression.
    steps = [b - a for a, b in zip(values, values[1:])]
    drift = statistics.fmean(steps)
    total = values[-1] - values[0]

    if spread >= config["volatile_sigma"]:
        return "VOLATILE"
    if abs(total) < config["deviation_absolute"]:
        return "STABLE"
    improving = drift > 0 if dimension not in INVERTED else drift < 0
    return "IMPROVING" if improving else "DECLINING"


def summarize(history, baseline=None, config=None):
    """Baseline + changes + per-dimension trends in one auditable object."""
    config_full = {**CONFIG, **(config or {})}
    baseline = baseline or compute_baseline(
        history, config, exclude_recent=config_full["recent_window"])
    changes = detect_changes(history, baseline, config)
    return {
        "baseline": baseline.to_dict(),
        "changes": [c.to_dict() for c in changes],
        "trends": {d: trend(history, d, config) for d in STATE_DIMENSIONS},
        "observations": len(history),
        "rule_version": RULE_VERSION,
        "is_diagnostic": False,
        "note": "Observed change from the patient's own baseline. Not an episode, "
                "not a diagnosis.",
    }
