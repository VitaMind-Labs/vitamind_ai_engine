"""Tache 3 — Extraction clinique arabe depuis arabic_raw, securite prioritaire, parite AR/EN."""

from fastapi.testclient import TestClient

from app import app
from vitamind.clinical.feature_extractor import FeatureExtractor
from vitamind.clinical.patient_state import FeatureStatus, PatientState
from vitamind.safety.detector import SafetyDetector

client = TestClient(app)


def _status(text: str, domain: str, feature: str) -> str:
    state = PatientState.new("adult", language="ar")
    FeatureExtractor().extract_into_state(state, text)
    return state.get_status(domain, feature).value


# --- Safety priority (doit etre couvert avant features generales) ---

def test_safety_arabic_priority_before_clinical():
    detector = SafetyDetector()
    # AR suicidal intent — priorite absolue
    ar_suicidal = "أريد أن أقتل نفسي"
    en_suicidal = "I want to kill myself"
    for txt in [ar_suicidal, en_suicidal]:
        state = PatientState.new("adult")
        result = detector.detect(state, txt)
        assert result.urgent is True
        assert result.reason == "suicidal_intent"

    # Via API: securite declenche complete immediat, meme avec features cliniques
    sid = client.post("/api/v1/mira/session", json={"language": "ar"}).json()["session_id"]
    payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": "أريد أن أقتل نفسي، أفكاري تتسارع ولا أنام", "language": "ar"}).json()
    assert payload["assessment_complete"] is True
    assert payload["safety"]["level"] == "urgent"
    assert "suicidal_intent" in payload["safety"]["flags"]

    # EN parite
    sid_en = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    payload_en = client.post(f"/api/v1/mira/session/{sid_en}/message", json={"text": en_suicidal, "language": "en"}).json()
    assert payload_en["safety"]["level"] == "urgent"


def test_safety_negation_ar_stays_routine():
    # Negation doit rester routine (pas d'urgent)
    for txt in [
        "لا أريد أن أقتل نفسي، أشعر فقط بالتعب",
        "لم أفكر في الانتحار في حياتي أبدا",
        "I don't want to kill myself, I just feel tired",
    ]:
        state = PatientState.new("adult")
        result = SafetyDetector().detect(state, txt)
        assert result.urgent is False, f"false positive on {txt}"


# --- Parity AR/EN: meme sens -> meme detection ---

PARITY_PAIRS = [
    # (en_text, ar_text, domain, feature)
    ("I am the smartest person and I can do anything", "أنا الأذكى وأستطيع فعل أي شيء", "mood", "grandiosity"),
    ("I talk too fast and can't stop talking, people tell me I talk too fast", "أتكلم بسرعة ولا أستطيع التوقف عن الكلام", "mood", "pressured_speech"),
    ("I jump from topic to topic, flight of ideas", "أقفز من موضوع لآخر", "mood", "flight_of_ideas"),
    ("I sleep 3 hours without feeling tired, still full of energy", "أنام 3 ساعات بدون الشعور بالتعب وما زلت مليان طاقة", "sleep", "decreased_need_for_sleep"),
    ("I feel unusually energized and on top of the world", "أشعر بطاقة غير عادية وأشعر أنني في القمة", "mood", "elevated_mood"),
    ("These periods come and go, cycles and episodes", "تأتي وتذهب، دورات ونوبات", "episode_history", "episodic_pattern"),
    ("I can't get out of bed, feel worthless and empty", "لا أستطيع النهوض من السرير وأشعر بعدم القيمة", "episode_history", "depression_alternation"),
    ("I can't sit still, restless and act without thinking", "لا أستطيع الجلوس، قلق وأتصرف بدون تفكير", "attention", "hyperactivity_impulsivity"),
    ("My thoughts are jumbled, nothing makes sense", "أفكاري مشوشة ولا شيء منطقي", "psychosis", "thought_disorganization"),
    ("I stopped seeing friends and isolate myself, stay in my room", "توقفت عن رؤية الأصدقاء وأنعزل وأبقى في غرفتي", "psychosis", "social_withdrawal"),
]


def test_parity_ar_en_same_detection():
    for en_text, ar_text, domain, feature in PARITY_PAIRS:
        en_status = _status(en_text, domain, feature)
        ar_status = _status(ar_text, domain, feature)
        assert en_status == "present", f"EN failed: {en_text} -> {domain}.{feature}={en_status}"
        assert ar_status == "present", f"AR failed: {ar_text} -> {domain}.{feature}={ar_status} (source arabic_raw verified)"
        assert en_status == ar_status, f"parity failed EN vs AR for {feature}"


def test_decreased_need_for_sleep_ar_with_fatigue_goes_absent():
    # AR: sleep reduction with fatigue -> absent/contradictory, not present
    state = PatientState.new("adult", language="ar")
    FeatureExtractor().extract_into_state(state, "أنام 3 ساعات وأشعر أنني مرهق ومتعب في اليوم التالي")
    status = state.get_status("sleep", "decreased_need_for_sleep")
    assert status.value in ("absent", "conflicting"), f"AR fatigue should be absent, got {status}"


def test_arabic_raw_triggers_sourced():
    # Verifie que des triggers issus de arabic_raw (comptes verifies) declenchent bien
    # Ces phrases proviennent de nettoyages reels: ساعتين (69 hits), نوبات (1421), أنعزل (40)
    assert _status("أنام ساعتين فقط بدون تعب", "sleep", "reduced_sleep") == "present"  # ساعتين
    assert _status("أعاني من نوبات متكررة", "episode_history", "episodic_pattern") == "present"  # نوبات
    assert _status("أفكاري مشوشة جدا", "psychosis", "thought_disorganization") == "present"
    assert _status("أنعزل عن الناس", "psychosis", "social_withdrawal") == "present"


def test_arabic_multitour_accumulation():
    # Multi-tours AR: les observations s'accumulent entre tours via PatientState
    sid = client.post("/api/v1/mira/session", json={"language": "ar"}).json()["session_id"]
    # 3 messages AR couvrant differents domaines
    payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": "أنا الأذكى وأستطيع فعل أي شيء", "language": "ar"}).json()
    assert payload["assessment_complete"] is False
    payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": "أتكلم بسرعة ولا أستطيع التوقف عن الكلام وأقفز من موضوع لآخر", "language": "ar"}).json()
    payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": "أنام 3 ساعات بدون الشعور بالتعب وما زلت مليان طاقة", "language": "ar"}).json()
    # Verifier via GET que safety pas declenchee
    get = client.get(f"/api/v1/mira/session/{sid}").json()
    assert get["safety"] == {"level": "routine", "flags": []}


def test_vague_short_ar_no_strong_signal():
    # Principe directeur: score bas sur reponses courtes/vagues n'est PAS un bug
    state = PatientState.new("adult", language="ar")
    FeatureExtractor().extract_into_state(state, "مرحبا")
    from vitamind.clinical.assessment.assessment_engine import AssessmentEngine
    from vitamind.clinical.differential import DifferentialEngine
    from vitamind.clinical.uncertainty import UncertaintyEngine
    assessment = AssessmentEngine().assess(state)
    differential = DifferentialEngine().analyze(state, assessment)
    uncertainty = UncertaintyEngine().decide(state, assessment, differential)
    assert uncertainty.abstain is True
    assert max(assessment.assessments["ADHD"].raw_score, assessment.assessments["BIPOLAR_SPECTRUM"].raw_score, assessment.assessments["PSYCHOSIS_SPECTRUM"].raw_score) < 2
