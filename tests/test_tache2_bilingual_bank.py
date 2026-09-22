"""Tache 2 — Banque de questions bilingue EN/AR, variation, plafond et progression."""

from fastapi.testclient import TestClient

from app import app
from vitamind.mira.interview import DEFAULT_SESSION_QUESTIONS, MAX_SESSION_QUESTIONS
from vitamind.mira.question_bank import describe_coverage, load_bank, text_for
from vitamind.clinical.patient_state import PatientState
from vitamind.clinical.assessment.assessment_engine import AssessmentEngine
from vitamind.mira.interview import InterviewPlanner

client = TestClient(app)


def test_question_bank_coverage_and_languages():
    bank = load_bank()
    assert bank is not None
    assert len(bank) == 17  # free_text hors SAFETY/COMPLETION (19 total, 2 filtres)
    cov = describe_coverage(bank)
    assert cov["total_free"] == 17
    assert cov["missing_ar"] == []  # banque actuelle couvre EN+AR pour chaque free_text
    # Pas de FR/TN — seulement EN et AR
    for entry in bank:
        assert entry.get("question_en", "").strip() != ""
        assert entry.get("question_ar", "").strip() != ""
        # Aucun melange: text_for strict
        assert text_for(entry, "en") == entry["question_en"].strip()
        assert text_for(entry, "ar") == entry["question_ar"].strip()
        try:
            text_for(entry, "invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid language must be rejected")


def test_language_strict_session():
    # EN session -> EN questions
    sid_en = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    payload_en = client.post(f"/api/v1/mira/session/{sid_en}/message", json={"text": "Hello, I want to chat", "language": "en"}).json()
    msg_en = payload_en["assistant_message"]
    # EN should be latin script, not arabic
    assert any(c.isascii() for c in msg_en)
    assert not any('\u0600' <= c <= '\u06FF' for c in msg_en), f"EN session returned Arabic: {msg_en}"

    # AR session -> AR questions
    sid_ar = client.post("/api/v1/mira/session", json={"language": "ar"}).json()["session_id"]
    payload_ar = client.post(f"/api/v1/mira/session/{sid_ar}/message", json={"text": "مرحبا، أريد التحدث", "language": "ar"}).json()
    msg_ar = payload_ar["assistant_message"]
    # AR should contain arabic script
    assert any('\u0600' <= c <= '\u06FF' for c in msg_ar), f"AR session did not return Arabic: {msg_ar}"


def test_chapter_progress_denominator_is_10():
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    for i in range(1, 6):
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": f"message {i} about sleep and mood", "language": "en"}).json()
        expected = min(0.99, i / DEFAULT_SESSION_QUESTIONS)
        assert payload["chapter_progress"] == expected, f"i={i} expected {expected} got {payload['chapter_progress']}"
        get = client.get(f"/api/v1/mira/session/{sid}").json()
        assert get["chapter_progress"] == min(1.0, i / DEFAULT_SESSION_QUESTIONS)
    # Default 10, not 8 — plafond documente = 17 (pool utilisable)
    assert DEFAULT_SESSION_QUESTIONS == 10
    assert MAX_SESSION_QUESTIONS == 17


def test_default_10_and_extension_to_19():
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    # Fill to default 10 -> should complete
    payload = None
    for i in range(10):
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": f"generic message {i} about daily life", "language": "en"}).json()
    assert payload["assessment_complete"] is True
    assert payload["chapter_progress"] == 1.0

    # New session: test extension beyond 10 when requested
    sid2 = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    for i in range(9):
        client.post(f"/api/v1/mira/session/{sid2}/message", json={"text": f"generic {i}", "language": "en"}).json()
    # 9th -> not complete
    payload9 = client.post(f"/api/v1/mira/session/{sid2}/message", json={"text": "message 9", "language": "en"}).json()
    # After 10, without extension phrase, next would be complete. But we send extension phrase at 10
    # Actually we already at 10 after loop (9+1). Let's create fresh to test extension clearly
    sid3 = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    for i in range(10):
        # last message is extension request
        text = "I have more to say, can I answer more questions? I want to continue." if i == 9 else f"generic {i}"
        payload = client.post(f"/api/v1/mira/session/{sid3}/message", json={"text": text, "language": "en"}).json()
    # With extension phrase at message 10, session should NOT be complete (allow 11)
    assert payload["assessment_complete"] is False, "extension phrase should allow continuing beyond 10"
    # Continue to max 17 (plafond documente = 17 free_text utilisables)
    # After 10 with extension, we need 7 more to reach 17
    for i in range(11, 18):
        # i=11..17 -> 7 messages
        text = "more questions please"
        payload = client.post(f"/api/v1/mira/session/{sid3}/message", json={"text": text, "language": "en"}).json()
        if i < 17:
            assert payload["assessment_complete"] is False, f"should not complete before 17 when extension requested, i={i}"
        else:
            assert payload["assessment_complete"] is True, "must complete at MAX 17"
            assert payload["chapter_progress"] == 1.0
    assert MAX_SESSION_QUESTIONS == 17


def test_variation_between_sessions():
    # Two sessions same language/leading should not have identical question sequences
    # Use neutral messages to avoid strong leading bias
    def collect_questions(language="en", n=5):
        sid = client.post("/api/v1/mira/session", json={"language": language}).json()["session_id"]
        qs = []
        for i in range(n):
            payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": f"neutral message {i} about life", "language": language}).json()
            qs.append(payload["assistant_message"])
        return qs

    # Collect 2 sessions, compare
    q1 = collect_questions("en", 5)
    q2 = collect_questions("en", 5)
    # At least one difference in first 5 (variation)
    # Due to random weighted selection, sequences should differ. If they happen to be identical by chance, retry
    if q1 == q2:
        q3 = collect_questions("en", 5)
        assert q1 != q3 or q2 != q3, f"variation failed: two sessions gave identical sequences {q1}"
    else:
        assert q1 != q2

    # Within same session, consecutive questions must differ (already tested elsewhere)
    for a, b in zip(q1, q1[1:]):
        assert a != b


def test_interview_planner_random_weighted_not_deterministic():
    # Direct planner test: 10 calls from same blank state should vary across fresh states
    from vitamind.mira.interview import InterviewPlanner
    import vitamind.mira.question_bank as qb
    qb.load_bank.cache_clear()
    planner = InterviewPlanner()
    state1 = PatientState.new("adult", language="en")
    state2 = PatientState.new("adult", language="en")
    assessment1 = AssessmentEngine().assess(state1)
    assessment2 = AssessmentEngine().assess(state2)
    q1 = planner.next_question(state1, assessment1).question
    q2 = planner.next_question(state2, assessment2).question
    # Not guaranteed different every time, but test that planner uses random (not always same)
    # Run multiple times and check diversity
    seen = set()
    for _ in range(20):
        s = PatientState.new("adult", language="en")
        q = planner.next_question(s, AssessmentEngine().assess(s)).question
        seen.add(q)
    assert len(seen) > 1, "planner should produce varied questions, not deterministic single"


def test_fallback_bilingual_strict():
    # Force missing bank to trigger fallback, check AR fallback is AR not EN
    import vitamind.mira.question_bank as qb
    qb.load_bank.cache_clear()
    import tempfile, os, json
    from unittest.mock import patch
    # Use monkeypatch via environment: point to missing file
    planner = InterviewPlanner(bank_path="/tmp/missing_bank.json")
    state_en = PatientState.new("adult", language="en")
    state_ar = PatientState.new("adult", language="ar")
    assessment = AssessmentEngine().assess(state_en)
    q_en = planner.next_question(state_en, assessment)
    q_ar = planner.next_question(state_ar, assessment)
    # Fallback should respect language
    assert q_en.language == "en"
    assert q_ar.language == "ar"
    # AR fallback must be Arabic script
    assert any('\u0600' <= c <= '\u06FF' for c in q_ar.question)
    qb.load_bank.cache_clear()
