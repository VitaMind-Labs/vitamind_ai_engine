"""T1 regression: Mira must not repeat the same question turn after turn."""
from fastapi.testclient import TestClient

from app import app
from vitamind.mira.interview import InterviewPlanner

client = TestClient(app)

RICH_MESSAGES = [
    "i want to discover mira",
    "I usually sleep very little, often just 2 to 4 hours, but I wake up wired and full of energy without feeling tired at all, and this happens most nights of the week.",
    "I sleep very little, often only 2 to 4 hours, yet I feel completely charged and I start many projects because my thoughts race all morning long every day.",
    "Even on short nights of three hours I feel rested and I take on a huge amount of work without any fatigue whatsoever during the day.",
    "My attention drifts at work and I forget what I was doing several times a day.",
]


def test_consecutive_questions_differ_over_many_turns():
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    questions = []
    for text in RICH_MESSAGES:
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": text, "language": "en"}).json()
        assert payload["assessment_complete"] is False
        questions.append(payload["assistant_message"])
    for previous, current in zip(questions, questions[1:]):
        assert previous != current, f"repeated question: {previous!r}"


def test_asked_slots_are_recorded_in_state():
    from vitamind.clinical.patient_state import PatientState
    from vitamind.clinical.assessment.assessment_engine import AssessmentEngine

    state = PatientState.new("adult")
    planner = InterviewPlanner()
    first = planner.next_question(state, AssessmentEngine().assess(state), chapter="MORNING")
    second = planner.next_question(state, AssessmentEngine().assess(state), chapter="MORNING")
    assert first.question != second.question
    asked = InterviewPlanner.asked_slots(state)
    assert len(asked) == 2
    assert asked[0] != asked[1]


def test_fallback_rotates_without_bank(tmp_path, monkeypatch):
    from vitamind.clinical.patient_state import PatientState
    from vitamind.clinical.assessment.assessment_engine import AssessmentEngine
    import vitamind.mira.question_bank as qb

    qb.load_bank.cache_clear()
    monkeypatch.setenv("MIRA_QUESTION_BANK", str(tmp_path / "missing.json"))
    try:
        state = PatientState.new("adult")
        planner = InterviewPlanner()
        seen = [planner.next_question(state, AssessmentEngine().assess(state)).question for _ in range(4)]
        for previous, current in zip(seen, seen[1:]):
            assert previous != current
    finally:
        qb.load_bank.cache_clear()
        monkeypatch.delenv("MIRA_QUESTION_BANK", raising=False)
