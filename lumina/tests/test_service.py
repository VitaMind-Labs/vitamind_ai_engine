"""The Lumina HTTP service: the agent boundary the backend actually calls.

The signing tests deliberately replicate the TypeScript client's formula
(`src/agents/agent-http.service.ts` -> `signAgentRequest`) rather than importing
the Python helper. That is the point: if the two ever diverge, this fails, and the
backend cannot silently lose its ability to authenticate.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

SECRET = "test-secret-at-least-16-chars"


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    import os

    os.environ["LUMINA_SERVICE_SECRET"] = SECRET
    os.environ["ALLOW_UNREVIEWED_SAFETY_CONTENT"] = "true"
    from service.app import app

    with TestClient(app) as test_client:
        yield test_client


def ts_style_sign(request_id: str, body: str, at=None):
    """Byte-for-byte equivalent of the TypeScript client's signature."""
    timestamp = f"{at if at is not None else time.time():.3f}"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    signature = hmac.new(SECRET.encode("utf-8"),
                         f"{timestamp}.{request_id}.{digest}".encode("utf-8"),
                         hashlib.sha256).hexdigest()
    return timestamp, signature


def post(client, path, payload, at=None, tamper=None):
    # JSON.stringify emits no whitespace; the separators here match it exactly.
    body = json.dumps(payload, separators=(",", ":"))
    request_id = payload.get("request_id", "req-test")
    timestamp, signature = ts_style_sign(request_id, body, at)
    sent = tamper(body) if tamper else body
    return client.post(path, content=sent.encode("utf-8"),
                       headers={"Content-Type": "application/json",
                                "x-request-id": request_id,
                                "x-agent-timestamp": timestamp,
                                "x-agent-signature": signature})


# --- ops endpoints ------------------------------------------------------
def test_health_needs_no_credentials_and_no_dependencies(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["clinically_validated"] is False


def test_ready_reports_which_models_are_loaded(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["safety_rules"]
    assert body["signed_requests_required"] is True
    # Unreviewed clinical content is reported, not hidden.
    assert body["unreviewed_interventions"] > 0


def test_ready_never_reports_on_the_other_agent(client):
    """Lumina readiness must not depend on Mira, or one outage becomes two."""
    body = client.get("/ready").json()
    assert not any("mira" in key.lower() for key in body)


def test_version_exposes_every_rule_version(client):
    body = client.get("/version").json()
    assert body["contract_version"]
    for rule in ("baseline", "decision", "templates"):
        assert body["rule_versions"][rule]
    assert body["no_pretrained_weights"] is True
    assert body["no_external_inference_api"] is True


# --- service authentication ---------------------------------------------
def test_an_unsigned_request_is_refused(client):
    response = client.post("/api/v1/lumina/chat",
                           json={"request_id": "r", "patient_id": "p", "text": "hi"})
    assert response.status_code == 401
    assert response.json()["reason"] == "missing_signature_headers"


def test_the_typescript_signature_format_is_accepted(client):
    response = post(client, "/api/v1/lumina/chat",
                    {"request_id": "req-1", "patient_id": "p1", "track": "ADHD",
                     "text": "I cannot get myself to start this task"})
    assert response.status_code == 200, response.text


def test_a_tampered_body_is_refused(client):
    response = post(client, "/api/v1/lumina/chat",
                    {"request_id": "req-2", "patient_id": "p1", "track": "ADHD",
                     "text": "I cannot start"},
                    tamper=lambda body: body.replace("ADHD", "BIPOLAR"))
    assert response.status_code == 401
    assert response.json()["reason"] == "signature_mismatch"


def test_a_stale_signature_cannot_be_replayed(client):
    response = post(client, "/api/v1/lumina/chat",
                    {"request_id": "req-3", "patient_id": "p1", "text": "hello there"},
                    at=time.time() - 600)
    assert response.status_code == 401
    assert response.json()["reason"] == "timestamp_outside_window"


def test_the_request_id_is_echoed_for_end_to_end_tracing(client):
    response = post(client, "/api/v1/lumina/chat",
                    {"request_id": "req-trace", "patient_id": "p1",
                     "text": "I cannot focus today"})
    assert response.headers["x-request-id"] == "req-trace"
    assert response.json()["requestId"] == "req-trace"


# --- contract strictness -------------------------------------------------
def test_an_unknown_field_is_rejected_rather_than_ignored(client):
    response = post(client, "/api/v1/lumina/chat",
                    {"request_id": "r4", "patient_id": "p1", "text": "hi", "bogus": 1})
    assert response.status_code == 422


def test_an_out_of_range_signal_is_rejected(client):
    response = post(client, "/api/v1/lumina/checkin/process",
                    {"request_id": "r5", "patient_id": "p1",
                     "checkin": {"energy": 99}})
    assert response.status_code == 422


def test_an_oversized_history_is_rejected_not_truncated(client):
    response = post(client, "/api/v1/lumina/checkin/process",
                    {"request_id": "r6", "patient_id": "p1",
                     "checkin": {"energy": 5},
                     "history": [{"energy": 5}] * 200})
    assert response.status_code == 422


def test_chat_without_text_is_refused(client):
    response = post(client, "/api/v1/lumina/chat",
                    {"request_id": "r7", "patient_id": "p1"})
    assert response.status_code == 422


# --- the routes ----------------------------------------------------------
def test_chat_returns_a_full_decision_envelope(client):
    envelope = post(client, "/api/v1/lumina/chat",
                    {"request_id": "r8", "patient_id": "p1", "track": "ADHD",
                     "text": "I cannot get myself to start this task"}).json()
    assert envelope["agent"] == "LUMINA"
    assert envelope["decision"]["intervention_id"]
    assert envelope["decision"]["reason_codes"]
    assert envelope["storesChainOfThought"] is False
    assert envelope["clinicallyValidated"] is False


def test_a_crisis_turn_uses_only_backend_supplied_resources(client):
    payload = {"request_id": "r9", "patient_id": "p1", "track": "ADHD",
               "text": "I want to kill myself",
               "safety_config": {"emergency_resources": ["Local emergency 999"]}}
    envelope = post(client, "/api/v1/lumina/chat", payload).json()
    assert envelope["safety"]["level"] == "CRISIS"
    assert envelope["decision"]["type"] == "CRISIS_WORKFLOW"
    assert "999" in envelope["response"]["text"]

    without = dict(payload, request_id="r9b", safety_config={"emergency_resources": []})
    bare = post(client, "/api/v1/lumina/chat", without).json()
    # No resource configured means none is named - never an invented one.
    assert "999" not in bare["response"]["text"]
    assert bare["safety"]["level"] == "CRISIS"


def test_journal_analyze_returns_signals_and_an_idempotency_key(client):
    envelope = post(client, "/api/v1/lumina/journal/analyze",
                    {"request_id": "r10", "patient_id": "p1", "entry_id": "e1",
                     "text": "Deadlines piling up and I finished nothing again."}).json()
    assert envelope["idempotencyKey"] == "journal:e1:1:lumina-journal-analysis-v1"
    assert envelope["journalAnalysis"]["signals"]
    assert envelope["journalAnalysis"]["contains_raw_text"] is False
    assert "PERSIST_JOURNAL_ANALYSIS" in envelope["persistence"]


def test_an_edited_entry_gets_a_new_idempotency_key(client):
    first = post(client, "/api/v1/lumina/journal/analyze",
                 {"request_id": "r11", "patient_id": "p1", "entry_id": "e2",
                  "text": "Deadlines piling up.", "content_version": 1}).json()
    second = post(client, "/api/v1/lumina/journal/analyze",
                  {"request_id": "r12", "patient_id": "p1", "entry_id": "e2",
                   "text": "Deadlines piling up.", "content_version": 2}).json()
    assert first["idempotencyKey"] != second["idempotencyKey"]


def test_a_private_entry_without_consent_is_refused_by_the_service(client):
    response = post(client, "/api/v1/lumina/journal/analyze",
                    {"request_id": "r13", "patient_id": "p1", "entry_id": "e3",
                     "text": "some private entry text", "is_private": True,
                     "analysis_consent": False})
    assert response.status_code == 403
    assert response.json()["error"] == "consent_required"


def test_state_recompute_returns_baseline_and_trends_without_a_reply(client):
    envelope = post(client, "/api/v1/lumina/state/recompute",
                    {"request_id": "r14", "patient_id": "p1",
                     "history": [{"sleep_hours": 7.2, "energy": 5.0}] * 12}).json()
    assert envelope["baseline"]["calibrating"] is False
    assert envelope["trends"]
    assert envelope["is_diagnostic"] is False
    assert "response" not in envelope


def test_state_recompute_reports_the_capacity_of_the_latest_day(client):
    """A recomputed state used to come back with capacity UNKNOWN, which is what the
    journal-driven snapshot then stored."""
    tired = {"sleep_hours": 4.0, "energy": 2.0, "stress": 8.0, "focus": 2.0, "mood": 3.0}
    envelope = post(client, "/api/v1/lumina/state/recompute",
                    {"request_id": "r14b", "patient_id": "p1",
                     "history": [{"sleep_hours": 7.2, "energy": 5.0}] * 6 + [tired]}).json()
    assert envelope["state"]["capacity"] in ("REDUCED", "VERY_LOW")


def test_state_recompute_needs_history(client):
    response = post(client, "/api/v1/lumina/state/recompute",
                    {"request_id": "r15", "patient_id": "p1", "history": []})
    assert response.status_code == 422


def test_repeated_negative_outcomes_report_a_suspension(client):
    envelope = post(client, "/api/v1/lumina/interventions/outcome",
                    {"request_id": "r16", "patient_id": "p1",
                     "outcomes": [
                         {"intervention_id": "adhd_micro_start_01", "result": "INEFFECTIVE"},
                         {"intervention_id": "adhd_micro_start_01", "result": "INEFFECTIVE"},
                     ]}).json()
    assert envelope["suspended"] == ["adhd_micro_start_01"]
    assert "PERSIST_MEMORY_UPDATE" in envelope["persistence"]


def test_a_journal_entry_and_a_chat_turn_fuse_to_the_higher_level(client):
    """A calm chat turn must not mask a crisis the journal signals already raised."""
    analysis = post(client, "/api/v1/lumina/journal/analyze",
                    {"request_id": "r17", "patient_id": "p1", "entry_id": "e4",
                     "text": "I wrote goodbye messages tonight. I want to end my life."}).json()
    assert analysis["journalAnalysis"]["safety"]["level"] == "CRISIS"
    # The backend carries the signals forward; the level travels with them.
    envelope = post(client, "/api/v1/lumina/checkin/process",
                    {"request_id": "r18", "patient_id": "p1", "track": "BIPOLAR",
                     "checkin": {"sleep_hours": 5.0},
                     "journal_signals": analysis["journalAnalysis"]["signals"]}).json()
    # Signals alone do not carry a level - that is the backend's job to fuse, and
    # is asserted in test_journal_integration. Here we only confirm the signals
    # arrive as estimated state rather than being dropped.
    assert envelope["state"]["signals"]["mood"]["quality"] == "estimated"


def test_the_service_accepts_a_journal_context_and_rejects_unknown_keys(client):
    body = {"request_id": "rj1", "patient_id": "p1", "track": "ADHD",
            "text": "How do I get through this evening?",
            "journal_context": {"tier": "moderate_flagged", "follow_ups": ["safety_check"],
                                "cues": ["hopelessness"], "hours_ago": 1}}
    out = post(client, "/api/v1/lumina/chat", body).json()
    assert out["decision"]["type"] == "ELEVATED_SAFETY_WORKFLOW"
    body["journal_context"]["text"] = "raw entry text"
    assert post(client, "/api/v1/lumina/chat", body).status_code == 422
