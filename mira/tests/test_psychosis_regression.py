"""Regression tests for Psychosis spectrum feature extraction and assessment pipeline.

Ensures that:
1. Full psychosis presentation (auditory perceptual experience + persecutory ideas +
   thought disorganization + social withdrawal) is correctly identified as
   psychosis_spectrum_clinical_assessment with HIGH match strength.
2. Psychosis presentation with hallucinations and thought disorganization (without
   explicit persecutory ideas) does NOT abstain and produces the psychosis pathway.
3. Negative denial ("never heard voices") correctly records absent.
"""

from fastapi.testclient import TestClient

from app import app
from vitamind.clinical.feature_extractor import FeatureExtractor
from vitamind.clinical.patient_state import FeatureStatus, PatientState
from vitamind.clinical.assessment.assessment_engine import AssessmentEngine
from vitamind.clinical.differential import DifferentialEngine
from vitamind.clinical.uncertainty import UncertaintyEngine

client = TestClient(app)

PSYCHOSIS_TRANSCRIPT = [
    "For the past few months, I've been hearing voices when I am completely alone in my room.",
    "The voices are whispering and telling me things that make me feel scared and unsafe.",
    "I feel like people are watching me and following me wherever I go, like they are out to get me.",
    "My thoughts are jumbled and I can't organize my thoughts anymore, nothing makes sense when I try to explain.",
    "I have stopped seeing friends and I completely isolate myself at home because I don't want to see anyone.",
    "I feel like strangers on the street are plotting against me and talking about me behind my back.",
    "My family says I've changed because I just stay in my room with the blinds closed all day.",
    "The confusion is constant and I don't know who or what to trust anymore.",
]


def test_psychosis_full_session_pathway_and_features():
    """Full psychosis transcript produces psychosis_spectrum_clinical_assessment with HIGH match."""
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    payload = None
    # Tache 2: session now 10 messages by default (was 8). Pad transcript to 10.
    extended = PSYCHOSIS_TRANSCRIPT + [
        "This withdrawal and perceptual pattern has lasted months and is getting worse.",
        "I want to know what clinical follow-up is recommended for these experiences.",
    ]
    for msg in extended:
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": msg, "language": "en"}).json()

    assert payload["assessment_complete"] is True
    result = payload["result"]

    assert result["recommended_pathway"] == "psychosis_spectrum_clinical_assessment"
    assert result["match_strength"] == "HIGH"
    assert result["condition_scores"]["psychosis"] >= 0.8
    assert result["condition_scores"]["psychosis"] > result["condition_scores"]["adhd"]
    assert result["condition_scores"]["psychosis"] > result["condition_scores"]["bipolar"]

    supporting = result["supporting_features"]
    assert "auditory_perceptual_experience" in supporting
    assert "persecutory_ideas" in supporting
    assert "thought_disorganization" in supporting
    assert "social_withdrawal" in supporting


def test_psychosis_without_paranoia_does_not_abstain():
    """Hallucinations + thought disorganization + social withdrawal without persecutory ideas."""
    state = PatientState.new(age_group="adult", age_years=22)
    extractor = FeatureExtractor()
    extractor.extract_into_state(state, "I hear voices when alone in the house.")
    extractor.extract_into_state(state, "My thoughts are jumbled and nothing makes sense.")
    extractor.extract_into_state(state, "I isolate myself and stopped seeing friends.")

    assert state.get_status("psychosis", "auditory_perceptual_experience") == FeatureStatus.PRESENT
    assert state.get_status("psychosis", "thought_disorganization") == FeatureStatus.PRESENT
    assert state.get_status("psychosis", "social_withdrawal") == FeatureStatus.PRESENT

    engine = AssessmentEngine()
    assessment = engine.assess(state)
    assert assessment.leading_condition == "PSYCHOSIS_SPECTRUM"
    assert assessment.assessments["PSYCHOSIS_SPECTRUM"].raw_score >= 3.0

    differential = DifferentialEngine().analyze(state, assessment)
    uncertainty = UncertaintyEngine().decide(state, assessment, differential)
    assert uncertainty.abstain is False
    assert uncertainty.reason == "prototype_evidence_is_coherent"


def test_psychosis_denial_rule():
    """Denial 'never heard voices' should yield absent status."""
    state = PatientState.new(age_group="adult")
    extractor = FeatureExtractor()
    extractor.extract_into_state(state, "I have never heard voices in my life.")
    assert state.get_status("psychosis", "auditory_perceptual_experience") == FeatureStatus.ABSENT
