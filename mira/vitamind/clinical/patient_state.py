"""Canonical, serializable patient memory used by every clinical component."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FeatureStatus(str, Enum):
    UNKNOWN = "unknown"
    PRESENT = "present"
    ABSENT = "absent"
    CONFLICTING = "conflicting"


class ObservationPolarity(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


class Chapter(str, Enum):
    MORNING = "MORNING"
    MIDDAY = "MIDDAY"
    EVENING = "EVENING"
    INNER_VOICE = "INNER_VOICE"
    SAFETY = "SAFETY"


@dataclass
class PatientContext:
    age_group: str
    age_years: int | None = None
    language: str = "auto"
    country_context: str | None = None


@dataclass
class Observation:
    domain: str
    feature: str
    polarity: ObservationPolarity
    confidence: float
    evidence_excerpt: str
    chapter: str | None = None
    message_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    superseded: bool = False


class PatientState:
    """Owns clinical memory; downstream engines only read or append to it."""

    def __init__(self, context: PatientContext, observations: list[Observation] | None = None):
        self.context = context
        self.observations = observations or []

    @classmethod
    def new(cls, age_group: str, age_years: int | None = None, language: str = "auto", country_context: str | None = None) -> "PatientState":
        return cls(PatientContext(age_group, age_years, language, country_context))

    def add_observation(self, domain: str, feature: str, polarity: ObservationPolarity | str, confidence: float, evidence_excerpt: str, chapter: Chapter | str | None = None, message_id: str | None = None) -> Observation:
        normalized_polarity = ObservationPolarity(polarity)
        bounded_confidence = max(0.0, min(1.0, float(confidence)))
        observation = Observation(domain, feature, normalized_polarity, bounded_confidence, evidence_excerpt, self._value(chapter), message_id)
        self.observations.append(observation)
        return observation

    def get_status(self, domain: str, feature: str) -> FeatureStatus:
        values = {item.polarity for item in self.observations if item.domain == domain and item.feature == feature and not item.superseded}
        if not values:
            return FeatureStatus.UNKNOWN
        if len(values) > 1:
            return FeatureStatus.CONFLICTING
        return FeatureStatus.PRESENT if ObservationPolarity.PRESENT in values else FeatureStatus.ABSENT

    def unknown_features(self, required: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return [(domain, feature) for domain, feature in required if self.get_status(domain, feature) == FeatureStatus.UNKNOWN]

    def resolve_feature(self, domain, feature, polarity, excerpt, message_id=None):
        """Explicit clarification supersedes prior evidence without deleting its audit trail."""
        if domain == 'safety':
            raise ValueError('Safety concerns cannot be cleared by symptom clarification.')
        for item in self.observations:
            if item.domain == domain and item.feature == feature:
                item.superseded = True
        return self.add_observation(domain, feature, polarity, 1, excerpt, 'CLARIFICATION', message_id)

    def safety_snapshot(self) -> dict[str, Any]:
        urgent_features = {"suicidal_intent", "homicidal_intent", "immediate_danger", "command_hallucination"}
        active = [item for item in self.observations if item.polarity == ObservationPolarity.PRESENT and item.domain == "safety" and item.feature in urgent_features]
        return {"urgent": bool(active), "active_features": [asdict(item) | {"polarity": item.polarity.value} for item in active]}

    def to_dict(self) -> dict[str, Any]:
        data = {"context": asdict(self.context), "observations": [asdict(item) for item in self.observations]}
        for item in data["observations"]:
            item["polarity"] = item["polarity"].value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "PatientState":
        data = json.loads(payload)
        context = PatientContext(**data["context"])
        observations = [Observation(**{**item, "polarity": ObservationPolarity(item["polarity"])}) for item in data.get("observations", [])]
        return cls(context, observations)

    @staticmethod
    def _value(value: Enum | str | None) -> str | None:
        return value.value if isinstance(value, Enum) else value
