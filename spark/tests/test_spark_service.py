"""Spark as a deployable agent: signed callers only, ADHD track only, stateless.

The signing helper below re-implements the backend's formula on purpose (same as
Lumina's service tests): if the TypeScript client and the verifier ever diverge,
this fails instead of the backend silently losing its ability to authenticate.
"""
import hashlib
import hmac
import json
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

SECRET = "test-secret-at-least-16-chars"
PATIENT = "ae3e23cc-5b26-468f-a8df-708519a4a144"
ROOT = Path(__file__).resolve().parents[1]


def organize_request(text="I need to email James tomorrow.", lang="EN", **inner):
    request = {"requestId": str(uuid4()),
               "patient": {"id": PATIENT, "language": lang},
               "message": {"text": text},
               "timeContext": {"referenceDate": "2026-09-29", "localTime": "14:30",
                               "availableMinutes": 20}}
    request.update(inner)
    return request


def envelope(track="ADHD", **kwargs):
    return {"contract_version": "spark-contract-1", "track": track,
            "request": organize_request(**kwargs)}


def signed(client, payload, path="/api/v1/spark/organize", at=None, tamper=None,
           request_id="req-test"):
    body = json.dumps(payload, separators=(",", ":"))
    timestamp = f"{at if at is not None else time.time():.3f}"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    signature = hmac.new(SECRET.encode("utf-8"),
                         f"{timestamp}.{request_id}.{digest}".encode("utf-8"),
                         hashlib.sha256).hexdigest()
    sent = tamper(body) if tamper else body
    return client.post(path, content=sent.encode("utf-8"), headers={
        "Content-Type": "application/json", "x-request-id": request_id,
        "x-agent-timestamp": timestamp, "x-agent-signature": signature})


@pytest.fixture(scope="module")
def client():
    import os

    os.environ["SPARK_SERVICE_SECRET"] = SECRET
    from spark_service.app import app

    with TestClient(app) as test_client:
        yield test_client


# --- authentication -----------------------------------------------------
def test_unsigned_requests_are_refused(client):
    response = client.post("/api/v1/spark/organize", json=envelope())
    assert response.status_code == 401
    assert response.json()["reason"] == "missing_signature_headers"


def test_tampered_body_is_refused(client):
    response = signed(client, envelope(), tamper=lambda b: b.replace("James", "Jamie"))
    assert response.status_code == 401
    assert response.json()["reason"] == "signature_mismatch"


def test_stale_signature_is_refused(client):
    response = signed(client, envelope(), at=time.time() - 3600)
    assert response.status_code == 401
    assert response.json()["reason"] == "timestamp_outside_window"


def test_ops_routes_need_no_signature(client):
    for path in ("/health", "/ready", "/version"):
        assert client.get(path).status_code == 200


# --- the ADHD-only rule -------------------------------------------------
def test_adhd_patient_is_served(client):
    response = signed(client, envelope("ADHD"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent"] == "SPARK" and body["track"] == "ADHD"
    assert body["contractVersion"] == "spark-contract-1"
    result = body["result"]
    assert result["module"] == "ADHD_EXECUTIVE_FUNCTION"
    assert result["plan"]["strategy"] in ("ONE_NEXT_ACTION", "SHORT_DAY_PLAN")
    assert result["taskOperations"][0]["operation"] == "ADD"
    assert result["taskOperations"][0]["status"] == "PROPOSED"  # the backend persists
    assert "PERSIST_TASK_OPERATIONS" in body["persistence"]


@pytest.mark.parametrize("track", ["BIPOLAR", "SCHIZOPHRENIA", "UNSPECIFIED"])
def test_other_tracks_are_refused(client, track):
    response = signed(client, envelope(track))
    assert response.status_code == 403
    assert response.json()["error"] == "spark_adhd_only"
    assert "result" not in response.json()


@pytest.mark.parametrize("track", ["adhd", "Adhd", "ADHD ", "PSYCHOSIS", "", 0, None, ["ADHD"]])
def test_track_match_is_exact(client, track):
    payload = envelope()
    payload["track"] = track
    response = signed(client, payload)
    assert response.status_code in (403, 422)
    assert "result" not in response.json()


def test_missing_track_is_not_defaulted_to_adhd(client):
    payload = envelope()
    del payload["track"]
    response = signed(client, payload)
    assert response.status_code == 422
    assert "result" not in response.json()


def test_inner_support_track_cannot_override_the_authoritative_track(client):
    payload = envelope("UNSPECIFIED")
    payload["request"]["patient"]["supportTrack"] = "ADHD"
    assert signed(client, payload).status_code == 403


def test_track_is_judged_before_the_payload_is_validated(client):
    """A non-ADHD caller learns nothing about the schema."""
    payload = {"contract_version": "spark-contract-1", "track": "BIPOLAR", "request": {"nope": 1}}
    assert signed(client, payload).status_code == 403


def test_other_tracks_never_reach_the_models(client, monkeypatch):
    from spark_service import app as service

    def boom(*args, **kwargs):
        raise AssertionError("organize() ran for a non-ADHD patient")

    monkeypatch.setattr(service._assistant, "organize", boom)
    assert signed(client, envelope("SCHIZOPHRENIA")).status_code == 403


# --- contract -----------------------------------------------------------
def test_contract_mismatch_is_rejected(client):
    payload = envelope()
    payload["contract_version"] = "spark-contract-0"
    assert signed(client, payload).status_code == 409


def test_unknown_fields_are_rejected(client):
    payload = envelope()
    payload["patient_condition"] = "ADHD"  # a stray, frontend-style field
    assert signed(client, payload).status_code == 422


def test_calendar_commit_is_not_supported_by_the_service(client):
    response = signed(client, envelope(calendar={"commit": True}))
    assert response.status_code == 422
    assert not (ROOT / "runtime_data").exists() or not list((ROOT / "runtime_data").glob("*.sqlite3"))


def test_validation_errors_never_echo_patient_text(client):
    payload = envelope(text="my secret plan")
    payload["request"]["patient"]["language"] = "FR"
    response = signed(client, payload)
    assert response.status_code == 422
    assert "my secret plan" not in response.text


def test_backend_supplied_tasks_are_used_and_service_holds_no_state(client):
    tasks = [{"temporaryId": "t1", "title": "Pay rent", "importance": "HIGH",
              "deadline": "2026-09-30"}]
    first = signed(client, envelope(text="Help me prioritize my schedule for today.",
                                    tasks=tasks)).json()["result"]
    assert first["plan"]["primaryTask"]["temporaryId"] == "t1"
    # A second call without the task knows nothing about it: no server-side memory.
    second = signed(client, envelope(text="Help me prioritize my schedule for today.")).json()["result"]
    assert second["plan"]["primaryTask"] is None


def test_completed_outcome_or_message_yields_operations_not_writes(client):
    tasks = [{"temporaryId": "t1", "title": "Finish the report"}]
    result = signed(client, envelope(text="I finished the report", tasks=tasks)).json()["result"]
    ops = result["taskOperations"]
    assert ops and ops[0]["operation"] == "COMPLETE" and ops[0]["status"] == "PROPOSED"


# --- localisation & safety ---------------------------------------------
def test_arabic_request_gets_an_arabic_reply(client):
    result = signed(client, envelope(text="لازم أكلم البنك", lang="AR")).json()["result"]
    assert result["language"] == "AR"
    assert any("؀" <= ch <= "ۿ" for ch in result["response"]["text"])


def test_crisis_text_is_a_safety_handoff_and_saves_nothing(client):
    body = signed(client, envelope(text="I want to kill myself and I need to buy milk")).json()
    assert body["result"]["safety"]["level"] == "CRISIS"
    assert body["result"]["plan"]["strategy"] == "SAFETY_HANDOFF"
    assert body["result"]["taskOperations"] == []
    assert "PERSIST_SAFETY_EVENT" in body["persistence"]
    assert "PERSIST_TASK_OPERATIONS" not in body["persistence"]


def test_lumina_safety_context_reaches_spark(client):
    """Lumina's recent journal analysis (via the backend) raises Spark's urgency."""
    journal = [{"date": "2026-09-29",
                "signals": {"safetyLevel": "ELEVATED", "overwhelm": True}}]
    result = signed(client, envelope(journalContext=journal)).json()["result"]
    assert result["safety"]["level"] == "ELEVATED"
    assert "PROVIDED_CURRENT_SAFETY_CONTEXT" in result["safety"]["flags"]
    assert result["plan"]["strategy"] == "SAFETY_HANDOFF"


def test_lumina_capacity_context_shapes_the_plan(client):
    reduced = signed(client, envelope(currentState={"capacity": "VERY_LOW"})).json()["result"]
    assert reduced["capacity"] == "VERY_LOW"


# --- readiness ----------------------------------------------------------
def test_ready_reports_every_model_and_the_shared_safety_copy(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["track"] == "ADHD"
    models = body["models"]
    assert models["intent_classifier"]["loaded"] and models["friction_classifier"]["loaded"]
    assert models["safety_journal_model"]["loaded"]
    assert models["safety_journal_model"]["version"] == "journal-linear-seed42-v1"


def test_spark_uses_luminas_journal_ai_not_a_copy():
    import journal_ai

    lumina_copy = (ROOT.parent / "lumina_agent" / "journal_ai").resolve()
    assert Path(journal_ai.__file__).resolve().parent == lumina_copy
    assert not (ROOT / "lumina" / "safety" / "journal_ai").exists()


def test_service_refuses_to_start_unsigned_by_omission(monkeypatch):
    from spark_service import app as service

    for name in ("SPARK_SERVICE_SECRET", "LUMINA_SERVICE_SECRET",
                 "SPARK_ALLOW_UNSIGNED", "LUMINA_ALLOW_UNSIGNED"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError):
        service.load_secret()
    with TestClient(service.app) as unconfigured:
        assert unconfigured.get("/ready").status_code == 503
        assert unconfigured.post("/api/v1/spark/organize", json=envelope()).status_code == 503
    monkeypatch.setenv("SPARK_ALLOW_UNSIGNED", "true")
    assert service.load_secret() is None
    # Leave the module configured for any test that runs after this one.
    monkeypatch.undo()
    monkeypatch.setenv("SPARK_SERVICE_SECRET", SECRET)
    service._load()


def test_falls_back_to_the_lumina_secret(monkeypatch):
    from spark_service import app as service

    monkeypatch.delenv("SPARK_SERVICE_SECRET", raising=False)
    monkeypatch.setenv("LUMINA_SERVICE_SECRET", "shared-agent-secret-value")
    assert service.load_secret() == "shared-agent-secret-value"
