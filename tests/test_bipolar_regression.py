"""Regression tests for Bipolar spectrum feature extraction and assessment.

Ensures that:
1. Full bipolar mania/depression presentation is correctly identified as
   mood_disorder_clinical_assessment with HIGH match strength and all key
   clinical features captured.
2. decreased_need_for_sleep is correctly extracted as 'present' when the user
   describes not needing sleep without fatigue (e.g., "don't feel tired at all").
3. Legitimate sleep reduction with fatigue ("sleep 3 hours but exhausted/tired")
   is correctly recognized as absent/contradictory for decreased_need_for_sleep.
"""

from fastapi.testclient import TestClient

from app import app
from vitamind.clinical.feature_extractor import FeatureExtractor
from vitamind.clinical.patient_state import FeatureStatus, PatientState
from vitamind.clinical.assessment.assessment_engine import AssessmentEngine
from vitamind.clinical.differential import DifferentialEngine
from vitamind.clinical.uncertainty import UncertaintyEngine
from vitamind.mira.agent import MiraAgent

client = TestClient(app)

BIPOLAR_TRANSCRIPT = [
    "Lately I've been feeling on top of the world, like I'm the smartest person in the room and I could do anything. I feel unusually high or energized, like nothing can stop me.",
    "I sleep maybe three or four hours a night, but I don't feel tired at all — I actually feel like I could run a marathon. I wake up full of energy and ready to take on the world.",
    "My mind is racing all the time. People tell me I talk too fast and jump from topic to topic. I start sentences and then move on to something completely different before I finish.",
    "I went on a spending spree last week and bought things I didn't need — thousands of dollars on gadgets and clothes. I also started many projects at work that I can't possibly finish.",
    "But then I crash into this deep depression where I can't get out of bed for days. Everything feels hopeless and I lose interest in everything I was excited about before.",
    "These periods come and go. The highs last about a week, then the lows hit for two or three weeks. This has been happening on and off for about five years now, since I was in my early twenties.",
    "It's destroying my relationships and my job. During the highs I make terrible decisions and during the lows I can barely function. My partner says I become a completely different person.",
    "Overall, I'd say the worst parts are the racing thoughts, the spending I can't control, and the fact that I barely sleep but still feel full of energy during those high periods. Then it all crashes and I feel worthless.",
]


def test_bipolar_full_session_pathway_and_features():
    """Full bipolar transcript produces mood_disorder_clinical_assessment with HIGH match."""
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    payload = None
    # Tache 2: session now 10 messages by default (was 8). Pad transcript to 10.
    extended = BIPOLAR_TRANSCRIPT + [
        "The same activation and crash pattern has repeated for years, affecting work and relationships.",
        "I need to understand how this episodic energy and mood shift should be followed up clinically.",
    ]
    for msg in extended:
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": msg, "language": "en"}).json()

    assert payload["assessment_complete"] is True
    result = payload["result"]

    assert result["recommended_pathway"] == "mood_disorder_clinical_assessment"
    assert result["match_strength"] == "HIGH"
    assert result["condition_scores"]["bipolar"] >= 0.8
    assert result["condition_scores"]["bipolar"] > result["condition_scores"]["adhd"]
    assert result["condition_scores"]["bipolar"] > result["condition_scores"]["psychosis"]

    supporting = result["supporting_features"]
    assert "decreased_need_for_sleep" in supporting
    assert "elevated_mood" in supporting
    assert "grandiosity" in supporting
    assert "racing_thoughts" in supporting
    assert "pressured_speech" in supporting
    assert "flight_of_ideas" in supporting
    assert "impulsive_spending" in supporting
    assert "increased_goal_directed_activity" in supporting
    assert "episodic_pattern" in supporting
    assert "depression_alternation" in supporting

    assert "decreased_need_for_sleep" not in result["contradictory_features"]


def test_decreased_need_for_sleep_negation_contexts():
    """Verify that 'don't feel tired' extracts as present, while 'exhausted and tired' extracts as absent."""
    extractor = FeatureExtractor()

    # Case 1: "don't feel tired at all" -> present
    state1 = PatientState.new(age_group="adult")
    extractor.extract_into_state(state1, "I only sleep 3 hours, but I don't feel tired at all.")
    assert state1.get_status("sleep", "decreased_need_for_sleep") == FeatureStatus.PRESENT

    # Case 2: "without feeling tired" -> present
    state2 = PatientState.new(age_group="adult")
    extractor.extract_into_state(state2, "I slept 3 hours without feeling tired.")
    assert state2.get_status("sleep", "decreased_need_for_sleep") == FeatureStatus.PRESENT

    # Case 3: "feel exhausted and tired" -> absent
    state3 = PatientState.new(age_group="adult")
    extractor.extract_into_state(state3, "I only sleep 3 hours and I feel exhausted and tired.")
    assert state3.get_status("sleep", "decreased_need_for_sleep") == FeatureStatus.ABSENT


def test_bipolar_uncertainty_does_not_abstain():
    """UncertaintyEngine should not abstain on a clear bipolar presentation."""
    state = PatientState.new(age_group="adult", age_years=28)
    extractor = FeatureExtractor()
    for msg in BIPOLAR_TRANSCRIPT:
        extractor.extract_into_state(state, msg)

    engine = AssessmentEngine()
    assessment = engine.assess(state)
    differential = DifferentialEngine().analyze(state, assessment)
    uncertainty = UncertaintyEngine().decide(state, assessment, differential)

    assert uncertainty.abstain is False
    assert uncertainty.reason == "prototype_evidence_is_coherent"
    assert uncertainty.confidence >= 0.75
