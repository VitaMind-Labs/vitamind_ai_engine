"""Non-diagnostic differential checks kept separate from target conditions."""

from dataclasses import dataclass, field

from .assessment.assessment_engine import AssessmentResult
from .patient_state import PatientState


@dataclass
class DifferentialResult:
    alternatives: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class DifferentialEngine:
    def analyze(self, state: PatientState, assessment: AssessmentResult) -> DifferentialResult:
        alternatives = ["anxiety", "depression", "sleep_deprivation", "substances_or_medications", "medical_or_neurologic"]
        notes = []
        if state.get_status("sleep", "reduced_sleep").value == "present" and state.get_status("sleep", "decreased_need_for_sleep").value != "present":
            notes.append("Reduced sleep with fatigue may explain concentration or mood changes.")
        return DifferentialResult(alternatives, notes)