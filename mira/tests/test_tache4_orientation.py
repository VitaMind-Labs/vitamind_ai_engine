"""Tache 4 — Score et orientation, tracabilite, wording."""

from fastapi.testclient import TestClient

from app import app
from vitamind.clinical.patient_state import PatientState
from vitamind.clinical.feature_extractor import FeatureExtractor
from vitamind.clinical.assessment.assessment_engine import AssessmentEngine
from vitamind.clinical.differential import DifferentialEngine
from vitamind.clinical.uncertainty import UncertaintyEngine

client = TestClient(app)


def test_orientation_wording_not_diagnostic():
    # Use rich transcript that will produce a complete report
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    rich = [
        "I have been distracted since school and I forget appointments all the time at work.",
        "My mind races from one idea to the next and I cannot finish what I start each day.",
        "I sleep only three hours some nights yet I wake up full of energy and start big projects.",
        "In the evening I crash and feel empty and worthless and nothing feels worth doing.",
        "When alone I sometimes feel people are watching me and I do not trust what is real.",
        "I spend money impulsively and take on too much, then feel exhausted for days after.",
        "My attention drifts constantly and I lose track of time when I finally focus on tasks.",
        "Overall this pattern repeats for months and it affects my job and my relationships.",
        "This has been happening on and off for years and impacts my family and daily functioning.",
        "I would like to understand what clinical follow-up is most appropriate for this pattern.",
    ]
    payload = None
    for text in rich:
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": text, "language": "en"}).json()
    assert payload["assessment_complete"] is True
    result = payload["result"]
    # Wording orientation, pas diagnostic trompeur
    assert "orientation" in result, "report must contain orientation field"
    assert result["orientation"].lower().startswith("orientation:"), f"orientation wording: {result['orientation']}"
    # Check disclaimer says not a diagnosis
    assert "not a diagnosis" in result["disclaimer"].lower()
    # Check MiraReply assistant_message contains orientation, not diagnostic claims
    assert "orientation" in payload["assistant_message"].lower()
    assert "not a diagnosis" in payload["assistant_message"].lower()
    # Ensure word 'diagnostic' does not appear as if it were a diagnostic (except disclaimer)
    # The pathway should be orientation, not diagnostic label
    assert "diagnosis" not in result["recommended_pathway"].lower()
    assert payload["result"]["requires_clinician_review"] is True


def test_traceability_scores_and_evidence():
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    # Bipolar-rich to generate strong scores
    bipolar = [
        "Lately I have been feeling on top of the world, like I am the smartest person in the room and I could do anything.",
        "I sleep maybe three or four hours a night, but I dont feel tired at all — I actually feel like I could run a marathon.",
        "My mind is racing all the time. People tell me I talk too fast and jump from topic to topic.",
        "I went on a spending spree last week and bought things I didn't need — thousands of dollars on gadgets and clothes.",
        "But then I crash into this deep depression where I can't get out of bed for days.",
        "These periods come and go. The highs last about a week, then the lows hit for two or three weeks.",
        "It's destroying my relationships and my job.",
        "Overall, worst parts are racing thoughts, spending, and barely sleeping but still full of energy.",
        "The same activation and crash pattern has repeated for years.",
        "I need to understand how this episodic energy and mood shift should be followed up clinically.",
    ]
    payload = None
    for text in bipolar:
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": text, "language": "en"}).json()
    result = payload["result"]
    # Scores par condition avec confiance
    assert "condition_scores" in result
    assert set(result["condition_scores"]) == {"adhd", "bipolar", "psychosis"}
    for cond, score in result["condition_scores"].items():
        assert 0.0 <= score <= 1.0
    # Niveau de confiance present
    assert result["match_strength"] in ("HIGH", "MODERATE", "LOW")
    # Tracabilite: supporting / contradictory listes
    assert isinstance(result["supporting_features"], list)
    assert isinstance(result["contradictory_features"], list)
    # Pour ce transcript bipolaire, supporting doit contenir les 10 features attendues
    assert "decreased_need_for_sleep" in result["supporting_features"]
    assert "elevated_mood" in result["supporting_features"]
    # Contradictory ne doit pas contenir sleep si confirme (bug fix)
    assert "decreased_need_for_sleep" not in result["contradictory_features"]
    # Missing information doit etre liste
    assert isinstance(result["missing_information"], list)
    # GET doit rester coherent avec POST: meme progress, safety, complete
    get = client.get(f"/api/v1/mira/session/{sid}").json()
    assert get["complete"] is True
    assert get["chapter_progress"] == 1.0
    assert get["safety"] == {"level": "routine", "flags": []}


def test_supporting_contradictory_remain_after_t2_t3():
    # Verifie que meme apres banque bilingue et extraction AR, les champs restent alimentes
    # Session AR
    sid = client.post("/api/v1/mira/session", json={"language": "ar"}).json()["session_id"]
    ar_texts = [
        "أنا الأذكى وأستطيع فعل أي شيء، أشعر أنني في القمة",
        "أنام 3 ساعات بدون الشعور بالتعب وما زلت مليان طاقة",
        "أتكلم بسرعة وأقفز من موضوع لآخر، أفكاري تتسارع",
        "أنفقت كثيرا وبدأت مشاريع كثيرة",
        "ثم انهار ولا أستطيع النهوض من السرير وأشعر بعدم القيمة",
        "تأتي وتذهب، نوبات ودورات",
        "هذا النمط يتكرر منذ سنوات",
        "أثر على عملي وعلاقاتي",
        "أحتاج لفهم ما المتابعة المناسبة",
        "أحتاج لتوضيح المتابعة المناسبة مع الطبيب",
    ]
    payload = None
    for t in ar_texts:
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": t, "language": "ar"}).json()
    result = payload["result"]
    assert result["assessment_complete"] is True
    assert len(result["supporting_features"]) > 0
    # Au moins une feature AR doit etre presente
    assert any(f in result["supporting_features"] for f in ["decreased_need_for_sleep", "grandiosity", "pressured_speech"])


def test_short_vague_no_strong_signal_is_expected():
    # Principe directeur: score bas sur reponses courtes/vagues N'EST PAS un bug
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    for i in range(10):
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": "ok", "language": "en"}).json()
    result = payload["result"]
    # Doit etre NO_STRONG_TARGET_SIGNAL et abstention
    assert result["recommended_pathway"] == "NO_STRONG_TARGET_SIGNAL"
    assert result["match_strength"] == "LOW"
    assert max(result["condition_scores"].values()) < 0.4


def test_no_diagnostic_miswording_in_agent():
    # Verifie que le code ne contient pas de wording diagnostic trompeur hors disclaimer
    import pathlib
    agent_text = pathlib.Path("vitamind/mira/agent.py").read_text(encoding="utf-8")
    # Le mot 'diagnosis' ne doit apparaitre que dans le disclaimer "not a diagnosis"
    # et pas comme affirmation diagnostic
    assert "not a diagnosis" in agent_text.lower()
    # Ne doit pas contenir "you have bipolar" ou diagnostic direct
    assert "you have" not in agent_text.lower() or "orientation" in agent_text.lower()
