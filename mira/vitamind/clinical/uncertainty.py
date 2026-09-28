"""Abstention policy: the prototype must be able to say that evidence is insufficient."""

from dataclasses import dataclass

from .assessment.assessment_engine import AssessmentResult
from .differential import DifferentialResult
from .patient_state import PatientState


@dataclass
class UncertaintyDecision:
    abstain: bool
    reason: str
    confidence: float


class UncertaintyEngine:
    def decide(self, state: PatientState, assessment: AssessmentResult, differential: DifferentialResult) -> UncertaintyDecision:
        scores = sorted((item.raw_score for item in assessment.assessments.values()), reverse=True)
        top = scores[0] if scores else 0.0
        margin = top - scores[1] if len(scores) > 1 else top
        abstain = top < 2 or margin < 1 or bool(state.safety_snapshot()["urgent"])
        reason = "insufficient_or_conflicting_evidence" if abstain else "prototype_evidence_is_coherent"
        return UncertaintyDecision(abstain, reason, round(min(1.0, top / 4), 3))