"""Composition root for deterministic clinical prototype assessments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..patient_state import FeatureStatus, PatientState


@dataclass
class ConditionAssessment:
    condition: str
    supporting_evidence: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    raw_score: float = 0.0


@dataclass
class AssessmentResult:
    assessments: dict[str, ConditionAssessment]
    leading_condition: str | None


def _score(state: PatientState, condition: str, features: tuple[tuple[str, str], ...]) -> ConditionAssessment:
    result = ConditionAssessment(condition)
    for domain, feature in features:
        status = state.get_status(domain, feature)
        if status == FeatureStatus.PRESENT:
            result.raw_score += 1
            result.supporting_evidence.append(feature)
        elif status == FeatureStatus.ABSENT:
            result.contradictory_evidence.append(feature)
        elif status == FeatureStatus.CONFLICTING:
            result.raw_score += 0.25
            result.contradictory_evidence.append(f"conflicting:{feature}")
        else:
            result.missing_information.append(feature)
    return result


class AssessmentEngine:
    def assess(self, state: PatientState) -> AssessmentResult:
        assessments = {
            "ADHD": _score(
                state,
                "ADHD",
                (
                    ("attention", "distractibility"),
                    ("attention", "forgetfulness"),
                    ("attention", "procrastination"),
                    ("attention", "hyperactivity_impulsivity"),
                    ("developmental_history", "childhood_onset"),
                ),
            ),
            "BIPOLAR_SPECTRUM": _score(
                state,
                "BIPOLAR_SPECTRUM",
                (
                    ("sleep", "decreased_need_for_sleep"),
                    ("mood", "elevated_mood"),
                    ("mood", "racing_thoughts"),
                    ("mood", "impulsive_spending"),
                    ("mood", "increased_goal_directed_activity"),
                    ("mood", "grandiosity"),
                    ("mood", "pressured_speech"),
                    ("mood", "flight_of_ideas"),
                    ("episode_history", "episodic_pattern"),
                    ("episode_history", "depression_alternation"),
                ),
            ),
            "PSYCHOSIS_SPECTRUM": _score(
                state,
                "PSYCHOSIS_SPECTRUM",
                (
                    ("psychosis", "auditory_perceptual_experience"),
                    ("psychosis", "persecutory_ideas"),
                    ("psychosis", "thought_disorganization"),
                    ("psychosis", "social_withdrawal"),
                ),
            ),
        }
        leading = max(assessments, key=lambda name: assessments[name].raw_score)
        return AssessmentResult(assessments, leading if assessments[leading].raw_score > 0 else None)