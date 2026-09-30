"""Lumina HTTP service.

The agent runs as its own process on its own port, behind HMAC service auth. It
holds no database connection and no per-patient state between requests: every
turn arrives with the authorized context the backend assembled for it, and leaves
as a structured envelope the backend validates and persists.

Endpoints mirror what the engine can actually do today. Anything the engine does
not implement is absent rather than stubbed, so a caller cannot build against a
route that returns a placeholder.
"""
from __future__ import annotations

import os
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from lumina import CONTRACT_VERSION
from lumina.baseline import RULE_VERSION as BASELINE_RULES
from lumina.capacity import estimate
from lumina.decision import RULE_VERSION as DECISION_RULES
from lumina.interventions import Outcome, OutcomeHistory
from lumina.memory import MemoryStore
from lumina.orchestrator import AGENT_VERSION, Lumina
from lumina.response import TEMPLATE_VERSION
from lumina.state import build_state
from lumina.text import VERSION as NORMALIZATION_VERSION

from .auth import (PUBLIC_PATHS, REQUEST_ID_HEADER, AuthError, load_secret,
                   verify)
from .contracts import (ContextBundle, JournalAnalyzeRequest, OutcomeRequest,
                        StateRecomputeRequest)

SERVICE = "lumina"
API = "/api/v1/lumina"

# Built once at startup. The engine is stateless per patient, so one instance
# serves every request without cross-contamination.
_engine: Lumina | None = None
_secret: str | None = None
_startup_error: str | None = None


def _load():
    global _engine, _secret, _startup_error
    try:
        _secret = load_secret()
    except RuntimeError as error:
        # Surfaced through /ready rather than crashing, so an orchestrator can
        # report *why* the service is unhealthy instead of a bare exit code.
        _startup_error = str(error)
        return
    allow_unreviewed = os.environ.get(
        "ALLOW_UNREVIEWED_SAFETY_CONTENT", "true").lower() == "true"
    try:
        _engine = Lumina.load(allow_unreviewed=allow_unreviewed)
        if not allow_unreviewed:
            _engine.catalog.assert_production_ready()
    except Exception as error:
        _startup_error = f"{type(error).__name__}: {error}"
        _engine = None


@asynccontextmanager
async def lifespan(_app):
    # A lifespan handler rather than the deprecated on_event hook: on_event does
    # not fire for a bare TestClient, which silently left the engine unloaded and
    # made every route answer 503 during testing.
    _load()
    yield


app = FastAPI(title="VitaMind Lumina agent", version=AGENT_VERSION,
              lifespan=lifespan,
              description="Longitudinal patient-support agent. Local models only: "
                          "no pretrained weights, no external inference API.")


@app.middleware("http")
async def _authenticate(request: Request, call_next):
    """Verify the signature on every non-public route, and echo the request id."""
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    body = await request.body()
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

    if _secret is not None:
        try:
            request_id = verify(_secret, request.headers, body)
        except AuthError as error:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "reason": error.reason},
                headers={REQUEST_ID_HEADER: request_id})

    request.state.request_id = request_id
    started = time.perf_counter()
    response: Response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers["x-processing-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response


def _require_engine():
    if _engine is None:
        raise RuntimeError(_startup_error or "engine not loaded")
    return _engine


def _history(entries):
    return [build_state(entry.model_dump(exclude_none=True)) for entry in entries]


def _memory_store(items):
    store = MemoryStore()
    for item in items:
        # Rejections are expected and harmless here: the store's own rules decide
        # what is worth carrying, and a refused item simply does not steer a turn.
        store.propose(item.category, item.content, item.source, key=item.key,
                      confidence=item.confidence,
                      justification=item.justification or "supplied by backend")
    return store


def _outcomes(items):
    history = OutcomeHistory()
    for item in items:
        history.record(Outcome(intervention_id=item.intervention_id,
                               result=item.result, engagement=item.engagement,
                               at=item.at))
    return history


# --- ops ----------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness only. No dependency is touched, so it stays fast and truthful."""
    return {"status": "ok", "service": SERVICE, "version": AGENT_VERSION,
            "clinically_validated": False}


@app.get("/ready")
def ready():
    """Readiness for this agent's own dependencies. Never checks Mira."""
    engine = _engine
    if engine is None:
        return JSONResponse(status_code=503, content={
            "status": "not_ready", "service": SERVICE, "reason": _startup_error})
    unreviewed = engine.catalog.unreviewed
    return {
        "status": "ready",
        "service": SERVICE,
        "safety_rules": engine.safety.rules_version,
        "safety_model": engine.safety.model.version if engine.safety.model else None,
        "understanding_model": engine.understanding.version if engine.understanding else None,
        "journal_analyzer": engine.journal is not None,
        # What each learned component does in the turn, so "loaded" is never
        # mistaken for "steering decisions".
        "models": {
            "safety_head": {
                "loaded": engine.safety.model is not None,
                "role": "escalation_only: fused with the journal rules as max(); cannot reach CRISIS"},
            "understanding_head": {
                "loaded": engine.understanding is not None,
                "role": "advisory: returned in the envelope, never read by the decision engine"},
            "intent_head": {
                "loaded": engine.intent_model is not None,
                "role": "disabled: abstains on every input; intent is routed by rules"},
            "journal_ai": {
                "loaded": engine.journal is not None,
                "role": "rules + journal classifier: sole path to CRISIS, journal signals"},
        },
        "intervention_catalog": engine.catalog.version,
        "unreviewed_interventions": len(unreviewed),
        "signed_requests_required": _secret is not None,
    }


@app.get("/version")
def version():
    return {
        "service": SERVICE,
        "agent_version": AGENT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "normalization": NORMALIZATION_VERSION,
        "rule_versions": {"baseline": BASELINE_RULES, "decision": DECISION_RULES,
                          "templates": TEMPLATE_VERSION},
        "no_pretrained_weights": True,
        "no_external_inference_api": True,
    }


# --- the agent ----------------------------------------------------------
def _turn(bundle: ContextBundle, request_class=None):
    engine = _require_engine()
    return engine.turn(
        text=bundle.text,
        checkin=bundle.checkin.model_dump(exclude_none=True) if bundle.checkin else None,
        history=_history(bundle.history),
        track=bundle.track,
        secondary_track=bundle.secondary_track,
        language=bundle.language,
        memories=_memory_store(bundle.memory),
        outcomes=_outcomes(bundle.intervention_outcomes),
        journal_signals=bundle.journal_signals,
        journal_context=(bundle.journal_context.model_dump()
                         if bundle.journal_context else None),
        resources=bundle.safety_config.emergency_resources,
        request_class=request_class or bundle.request_class,
        request_id=bundle.request_id,
    )


@app.post(API + "/chat")
def chat(bundle: ContextBundle):
    if not bundle.text:
        return JSONResponse(status_code=422, content={
            "error": "invalid_request", "reason": "chat requires text"})
    return _turn(bundle, "CHAT_SHORT" if bundle.request_class not in
                 ("CHAT_SHORT", "CHAT_DEEP") else bundle.request_class)


@app.post(API + "/checkin/process")
def process_checkin(bundle: ContextBundle):
    if bundle.checkin is None:
        return JSONResponse(status_code=422, content={
            "error": "invalid_request", "reason": "checkin/process requires a checkin"})
    return _turn(bundle, "CHECKIN")


@app.post(API + "/journal/analyze")
def analyze_journal(payload: JournalAnalyzeRequest):
    engine = _require_engine()
    try:
        return engine.analyze_journal(
            payload.text, entry_id=payload.entry_id, language=payload.language,
            content_version=payload.content_version,
            is_private=payload.is_private,
            analysis_consent=payload.analysis_consent,
            request_id=payload.request_id)
    except PermissionError as error:
        return JSONResponse(status_code=403, content={
            "error": "consent_required", "reason": str(error)})


@app.post(API + "/state/recompute")
def recompute_state(payload: StateRecomputeRequest):
    """Baseline, changes and trends from a signal series. No response text."""
    from lumina.baseline import summarize
    history = _history(payload.history)
    if not history:
        return JSONResponse(status_code=422, content={
            "error": "invalid_request", "reason": "state/recompute requires history"})
    result = summarize(history)
    latest = history[-1]
    # No free text here, so no safety verdict: capacity is estimated from the
    # signals alone (a crisis reading only ever comes from a chat or journal turn).
    latest.capacity = estimate(latest).level
    return {"requestId": payload.request_id, "agent": "LUMINA",
            "agentVersion": AGENT_VERSION, "contractVersion": CONTRACT_VERSION,
            "state": latest.to_dict(), **result,
            "persistence": ["PERSIST_INTERACTION"]}


@app.post(API + "/interventions/outcome")
def intervention_outcome(payload: OutcomeRequest):
    """Evaluate recorded outcomes into suspensions and memory updates."""
    history = _outcomes(payload.outcomes)
    suspended = sorted(history.suspended())
    return {
        "requestId": payload.request_id,
        "agent": "LUMINA",
        "agentVersion": AGENT_VERSION,
        "contractVersion": CONTRACT_VERSION,
        "suspended": suspended,
        "successRates": {o.intervention_id: history.success_rate(o.intervention_id)
                         for o in payload.outcomes},
        "memoryUpdates": history.to_memory_updates(),
        "persistence": ["PERSIST_OUTCOME"] +
                       (["PERSIST_MEMORY_UPDATE"] if suspended else []),
    }


@app.exception_handler(RuntimeError)
def _engine_unavailable(request: Request, error: RuntimeError):
    # A missing engine is a service fault, not a client error, and must never be
    # answered with something that looks like a successful, safe result.
    return JSONResponse(status_code=503, content={
        "error": "engine_unavailable", "reason": str(error)})


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("LUMINA_HOST", "127.0.0.1"),
                port=int(os.environ.get("LUMINA_PORT", "8102")))


if __name__ == "__main__":
    main()
