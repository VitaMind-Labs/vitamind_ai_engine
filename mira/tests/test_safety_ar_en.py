"""T3: AR/EN safety parity. Every urgent EN motif must fire identically in AR (MSA)."""
from fastapi.testclient import TestClient

from app import app
from vitamind.clinical.patient_state import PatientState
from vitamind.safety.detector import SafetyDetector

client = TestClient(app)


def check(text):
    return SafetyDetector().detect(PatientState.new("adult"), text)


def live_level(text, language="ar"):
    sid = client.post("/api/v1/mira/session", json={"language": language}).json()["session_id"]
    payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": text, "language": language}).json()
    client.delete(f"/api/v1/mira/session/{sid}")
    return payload["safety"]["level"], payload["safety"]["flags"], payload["assessment_complete"]


def test_required_pair_identical():
    en = check("I want to kill myself.")
    ar = check("أريد أن أقتل نفسي")
    assert (en.urgent, en.reason) == (True, "suicidal_intent")
    assert (ar.urgent, ar.reason) == (True, "suicidal_intent")
    assert live_level("I want to kill myself.", "en")[0] == "urgent"
    assert live_level("أريد أن أقتل نفسي", "ar")[0] == "urgent"


AFFIRMATIVE_PAIRS = [
    ("I will end my life tonight.", "سأنهي حياتي الليلة", "suicidal_intent"),
    ("I am thinking about suicide every day.", "أفكر في الانتحار كل يوم", "suicidal_intent"),
    ("I want to hurt someone at work.", "أريد أن أؤذي شخصا في العمل", "homicidal_intent"),
    ("I am in immediate danger, please help.", "أنا في خطر فوري، ساعدوني", "immediate_danger"),
    ("I am not safe right now and I am scared.", "لست آمنا الآن وأنا خائف", "immediate_danger"),
    ("Voices are telling me to hurt myself.", "أنا أسمع أصواتا تأمرني أن أؤذي نفسي", "suicidal_intent"),
]


def test_affirmative_pairs_fire_identically():
    for en_text, ar_text, reason in AFFIRMATIVE_PAIRS:
        en = check(en_text)
        ar = check(ar_text)
        assert en.urgent and en.reason == reason, en_text
        assert ar.urgent and ar.reason == reason, ar_text


NEGATED_CALM = [
    "I don't want to kill myself, I just feel tired.",
    "I have never thought about suicide in my life.",
    "لا أريد أن أقتل نفسي، أشعر فقط بالتعب.",
    "لم أفكر في الانتحار في حياتي أبدا.",
    "I do not want to hurt myself or anyone else.",
]


def test_denials_stay_routine():
    for text in NEGATED_CALM:
        result = check(text)
        assert result.urgent is False, text


def test_no_prefix_does_not_silence_urgent():
    # "No. I want to kill myself." must stay urgent (bare "no" is not denial).
    result = check("No. I want to kill myself and I mean it.")
    assert result.urgent is True
