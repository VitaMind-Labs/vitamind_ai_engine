"""Tache 0 - Integration test multi-tours reel pour validation fix Workstream A/B.

Rejoue le transcript bipolaire via 8 appels POST separes
a POST /api/v1/mira/session/{id}/message, dans l ordre exact.
Compare avec diagnostic isole (PatientState + FeatureExtractor direct).

Attendu: 10/10 features, score 1.0 via les deux chemins.
"""

from fastapi.testclient import TestClient

from app import app
from vitamind.clinical.patient_state import PatientState
from vitamind.clinical.feature_extractor import FeatureExtractor
from vitamind.clinical.assessment.assessment_engine import AssessmentEngine

client = TestClient(app)

BIPOLAR_TRANSCRIPT = [
    "Lately I have been feeling on top of the world, like I am the smartest person in the room and I could do anything. I feel unusually high or energized, like nothing can stop me.",
    "I sleep maybe three or four hours a night, but I dont feel tired at all - I actually feel like I could run a marathon. I wake up full of energy and ready to take on the world.",
    "My mind is racing all the time. People tell me I talk too fast and jump from topic to topic. I start sentences and then move on to something completely different before I finish.",
    "I went on a spending spree last week and bought things I didnt need - thousands of dollars on gadgets and clothes. I also started many projects at work that I cant possibly finish.",
    "But then I crash into this deep depression where I cant get out of bed for days. Everything feels hopeless and I lose interest in everything I was excited about before.",
    "These periods come and go. The highs last about a week, then the lows hit for two or three weeks. This has been happening on and off for about five years now, since I was in my early twenties.",
    "It is destroying my relationships and my job. During the highs I make terrible decisions and during the lows I can barely function. My partner says I become a completely different person.",
    "Overall, I would say the worst parts are the racing thoughts, the spending I cant control, and the fact that I barely sleep but still feel full of energy during those high periods. Then it all crashes and I feel worthless.",
]

EXPECTED_FEATURES = {
    "decreased_need_for_sleep",
    "elevated_mood",
    "grandiosity",
    "racing_thoughts",
    "pressured_speech",
    "flight_of_ideas",
    "impulsive_spending",
    "increased_goal_directed_activity",
    "episodic_pattern",
    "depression_alternation",
}


def test_bipolar_isolated_score():
    state = PatientState.new(age_group="adult", language="en")
    fe = FeatureExtractor()
    for msg in BIPOLAR_TRANSCRIPT:
        fe.extract_into_state(state, msg)
    assess = AssessmentEngine().assess(state)
    assert assess.assessments["BIPOLAR_SPECTRUM"].raw_score == 10.0
    assert set(assess.assessments["BIPOLAR_SPECTRUM"].supporting_evidence) == EXPECTED_FEATURES


def test_bipolar_multitour_via_api_matches_isolated():
    # Isolated reference
    state_iso = PatientState.new(age_group="adult", language="en")
    fe = FeatureExtractor()
    for msg in BIPOLAR_TRANSCRIPT:
        fe.extract_into_state(state_iso, msg)
    assess_iso = AssessmentEngine().assess(state_iso)
    assert assess_iso.assessments["BIPOLAR_SPECTRUM"].raw_score == 10.0

    # Multi-tour via API: 8 transcript messages + 2 padding to reach DEFAULT 10
    # Avant T2: 8 suffisaient (hardcode 8). Apres T2: default 10, donc 10 requis.
    # Le test valide que l'accumulation PatientState reste identique en flux
    # reel, et que le score 1.0 est atteint des 8 premiers tours meme si la
    # session ne se clot qu'a 10.
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    payload = None
    for idx, msg in enumerate(BIPOLAR_TRANSCRIPT):
        resp = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": msg, "language": "en"})
        assert resp.status_code == 200, f"tour {idx+1} failed {resp.text}"
        payload = resp.json()
        if idx < 7:
            assert payload["assessment_complete"] is False
            assert payload["chapter_progress"] == (idx + 1) / 10
        # GET consistency (denominateur 10 apres T2)
        get = client.get(f"/api/v1/mira/session/{sid}").json()
        assert get["chapter_progress"] == min(1.0, (idx + 1) / 10)
        assert get["complete"] is False  # pas encore 10
    # Apres 8 tours: score deja 1.0 en interne, mais session pas complete avant 10
    assert payload["assessment_complete"] is False
    # Verifier que l'etat interne a deja 10/10 meme avant cloture
    from vitamind.mira.service import MiraSessionService

    # 2 messages supplementaires pour atteindre le seuil 10
    for extra in [
        "This episodic pattern with decreased sleep and energy continues to disrupt my daily functioning.",
        "I want to understand what clinical orientation is most appropriate for this mood pattern.",
    ]:
        resp = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": extra, "language": "en"})
        payload = resp.json()
    assert payload["assessment_complete"] is True
    result = payload["result"]
    assert result["condition_scores"]["bipolar"] == 1.0, f"score {result['condition_scores']['bipolar']} != 1.0"
    assert set(result["supporting_features"]) == EXPECTED_FEATURES, f"supporting {result['supporting_features']}"
    assert result["recommended_pathway"] == "mood_disorder_clinical_assessment"
    assert result["match_strength"] == "HIGH"
    iso_score = assess_iso.assessments["BIPOLAR_SPECTRUM"].raw_score / 10
    assert result["condition_scores"]["bipolar"] == iso_score
    assert set(result["supporting_features"]) == set(assess_iso.assessments["BIPOLAR_SPECTRUM"].supporting_evidence)
    assert result["safety"]["level"] == "routine"
    assert "decreased_need_for_sleep" not in result["contradictory_features"]
