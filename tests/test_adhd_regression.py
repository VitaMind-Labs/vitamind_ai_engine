"""Regression tests for ADHD feature extraction and assessment pipeline.

Ensures that:
1. Classic ADHD presentation (inattention + hyperactivity/impulsivity + childhood onset)
   is correctly identified as ADHD_focused_clinical_assessment with HIGH match strength.
2. Predominantly inattentive presentation (without hyperactivity) still scores HIGH.
3. Key clinical features (distractibility, forgetfulness, procrastination,
   hyperactivity_impulsivity, childhood_onset) are captured in supporting_features.
"""

from fastapi.testclient import TestClient

from app import app
from vitamind.clinical.feature_extractor import FeatureExtractor
from vitamind.clinical.patient_state import FeatureStatus, PatientState
from vitamind.clinical.assessment.assessment_engine import AssessmentEngine
from vitamind.clinical.differential import DifferentialEngine
from vitamind.clinical.uncertainty import UncertaintyEngine

client = TestClient(app)

ADHD_TRANSCRIPT = [
    "I've struggled with being distracted and losing focus ever since I was a child at school.",
    "I constantly forget appointments and forget things I was just about to do.",
    "I put things off until the last minute and procrastinate horribly on any paperwork or routine tasks.",
    "I can't sit still in meetings, I'm always restless and fidget with my hands or pens.",
    "I make impulsive decisions and act without thinking about the consequences.",
    "My mind feels like it has ten channels playing at once, and my attention drifts constantly.",
    "People at work complain that I lose track of important deadlines and don't finish tasks.",
    "This has been my entire life since childhood, and it's making it impossible to manage my daily responsibilities.",
]


def test_adhd_full_session_pathway_and_features():
    """Full ADHD transcript produces ADHD_focused_clinical_assessment with HIGH match."""
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    payload = None
    # Tache 2: session now 10 messages by default (was 8). Pad transcript to 10.
    extended = ADHD_TRANSCRIPT + [
        "Even with reminders I still miss deadlines and feel overwhelmed every week.",
        "I want to understand if this long-standing pattern needs clinical follow-up.",
    ]
    for msg in extended:
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": msg, "language": "en"}).json()

    assert payload["assessment_complete"] is True
    result = payload["result"]

    assert result["recommended_pathway"] == "ADHD_focused_clinical_assessment"
    assert result["match_strength"] == "HIGH"
    assert result["condition_scores"]["adhd"] >= 0.8
    assert result["condition_scores"]["adhd"] > result["condition_scores"]["bipolar"]
    assert result["condition_scores"]["adhd"] > result["condition_scores"]["psychosis"]

    supporting = result["supporting_features"]
    assert "childhood_onset" in supporting
    assert "distractibility" in supporting
    assert "forgetfulness" in supporting
    assert "procrastination" in supporting
    assert "hyperactivity_impulsivity" in supporting


def test_adhd_inattentive_subtype_still_qualifies():
    """Patient with inattentive symptoms (no hyperactivity) should still reach ADHD pathway."""
    state = PatientState.new(age_group="adult", age_years=26)
    extractor = FeatureExtractor()
    extractor.extract_into_state(state, "Since school I have had trouble concentrating and difficulty focusing.")
    extractor.extract_into_state(state, "I lose things and forget appointments constantly.")
    extractor.extract_into_state(state, "I procrastinate on everything.")

    assert state.get_status("developmental_history", "childhood_onset") == FeatureStatus.PRESENT
    assert state.get_status("attention", "distractibility") == FeatureStatus.PRESENT
    assert state.get_status("attention", "forgetfulness") == FeatureStatus.PRESENT
    assert state.get_status("attention", "procrastination") == FeatureStatus.PRESENT

    engine = AssessmentEngine()
    assessment = engine.assess(state)
    assert assessment.leading_condition == "ADHD"
    assert assessment.assessments["ADHD"].raw_score >= 4.0

    differential = DifferentialEngine().analyze(state, assessment)
    uncertainty = UncertaintyEngine().decide(state, assessment, differential)
    assert uncertainty.abstain is False
