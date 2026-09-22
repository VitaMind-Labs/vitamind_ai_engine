from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_and_session_lifecycle():
    assert client.get("/health").json()["status"] == "ok"
    started = client.post("/api/v1/mira/session", json={"language": "en"}).json()
    session_id = started["session_id"]
    response = client.post(f"/api/v1/mira/session/{session_id}/message", json={"text": "I have been distracted since school.", "language": "en"})
    assert response.status_code == 200
    assert response.json()["session_id"] == session_id
    assert client.get(f"/api/v1/mira/session/{session_id}").status_code == 200
    assert client.delete(f"/api/v1/mira/session/{session_id}").json() == {"success": True}


def test_unknown_session_is_not_an_internal_error():
    response = client.post("/api/v1/mira/session/missing/message", json={"text": "hello", "language": "en"})
    assert response.status_code == 404


def test_completed_message_matches_frontend_report_contract():
    started = client.post("/api/v1/mira/session", json={"language": "en"}).json()
    session_id = started["session_id"]
    response = None
    # Tache 2: default now 10 (was 8)
    for index in range(10):
        response = client.post(f"/api/v1/mira/session/{session_id}/message", json={"text": f"I have been distracted since school, message {index}.", "language": "en"})
    payload = response.json()
    assert payload["assessment_complete"] is True
    assert set(payload["result"]["condition_scores"]) == {"adhd", "bipolar", "psychosis"}
    assert payload["result"]["safety"]["flags"] == []


def test_urgent_message_returns_a_complete_safety_report():
    started = client.post("/api/v1/mira/session", json={"language": "en"}).json()
    payload = client.post(f"/api/v1/mira/session/{started['session_id']}/message", json={"text": "I am going to kill myself.", "language": "en"}).json()
    assert payload["assessment_complete"] is True
    assert payload["safety"]["level"] == "urgent"
    assert payload["safety"]["flags"] == ["suicidal_intent"]


def test_invalid_languages_are_rejected_at_the_api_boundary():
    for language in ["fr", "derja", "tn", "xx", ""]:
        response = client.post("/api/v1/mira/session", json={"language": language})
        assert response.status_code == 400