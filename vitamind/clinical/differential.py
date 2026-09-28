"""Record reported possible confounders; do not invent alternative diagnoses."""
import re
from dataclasses import dataclass, field
from .language import patient_clauses, negated_before

@dataclass
class DifferentialResult:
    alternatives: list = field(default_factory=list)
    notes: list = field(default_factory=list)

class DifferentialEngine:
    def analyze(self,state,assessment=None):
        found=set()
        patterns={'medication_context':r'\b(?:medicine|medication|prescription|stimulant)\b|(?:دواء|ادوية)',
                  'substance_context':r'\b(?:alcohol|cannabis|caffeine|drugs)\b|(?:كحول|مخدر|كافيين)',
                  'medical_context':r'\b(?:thyroid|illness|medical condition)\b|(?:الغدة|مرض جسدي)'}
        for obs in state.observations:
            if obs.domain!='context' or obs.superseded: continue
            for clause,own in patient_clauses(obs.evidence_excerpt):
                if not own: continue
                for label,pattern in patterns.items():
                    if any(not negated_before(clause,m.start()) for m in re.finditer(pattern,clause)):
                        found.add(label)
        if state.get_status('sleep','reduced_sleep').value=='present' and state.get_status('sleep','decreased_need_for_sleep').value=='absent':
            found.add('sleep_loss_with_fatigue')
        return DifferentialResult(sorted(found),['Possible contributing context reported; clinician interpretation needed.'] if found else [])
