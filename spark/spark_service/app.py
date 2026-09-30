"""Spark HTTP service: Lumina's ADHD specialist, on its own port.

Same boundary rules as the Lumina agent, because it sits behind the same backend:

* every non-public route is HMAC-signed with the shared agent secret (Lumina's own
  verifier, loaded from `lumina/service/auth.py`), so only the backend can call it;
* the agent holds no database and no per-patient state. The backend sends the
  patient's open tasks with each request and applies the task operations that
  come back, so the assistant's memory is the backend's database, not a SQLite
  file on the agent's disk;
* access is ADHD-only. The backend resolves the patient's authoritative track
  (clinician diagnosis, then Mira orientation) and sends it as `track`; anything
  but ADHD is refused before any model runs. This is a second gate, not the
  first: the backend refuses non-ADHD patients before it ever calls Spark.

Local-development standalone API (unsigned, loopback, no track check) still lives
in `lumina/api.py`; this module is the deployable one.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import Body, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from lumina.adhd.executive_function.errors import AssistantError
from lumina.adhd.executive_function.orchestrator import LuminaADHD
from lumina.shared import service_auth

from . import CONTRACT_VERSION, SPARK_VERSION
from .contracts import SPARK_TRACK, SparkOrganizeRequest

SERVICE = "spark"
API = "/api/v1/spark"
AGENT = "SPARK"

_auth = service_auth()
logger = logging.getLogger("spark.service")

_assistant: LuminaADHD | None = None
_secret: str | None = None
_startup_error: str | None = None


def load_secret():
    """The shared agent secret. Spark accepts its own name, then Lumina's.

    One backend signs both agents, so a single-secret deployment is the normal
    case. Running unsigned is a local-development choice that has to be made
    explicitly, never by omission.
    """
    for name in ("SPARK_SERVICE_SECRET", "LUMINA_SERVICE_SECRET"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if any(os.environ.get(name, "false").lower() == "true"
           for name in ("SPARK_ALLOW_UNSIGNED", "LUMINA_ALLOW_UNSIGNED")):
        return None
    raise RuntimeError(
        "SPARK_SERVICE_SECRET (or LUMINA_SERVICE_SECRET) is not set. Set it, or set "
        "SPARK_ALLOW_UNSIGNED=true for local development only - never in production.")


def _load():
    global _assistant, _secret, _startup_error
    # A reload must start from nothing: a stale assistant from an earlier start
    # would keep answering after its configuration had been withdrawn.
    _assistant, _secret, _startup_error = None, None, None
    try:
        _secret = load_secret()
    except RuntimeError as error:
        # Reported through /ready rather than crashing, like the other agents.
        _startup_error = str(error)
        return
    try:
        # calendar=None: stateless. The backend persists tasks.
        _assistant = LuminaADHD(calendar=None)
    except Exception as error:  # a corrupt model artifact must surface, not vanish
        _startup_error = f"{type(error).__name__}: {error}"
        _assistant = None


@asynccontextmanager
async def lifespan(_app):
    _load()
    yield


app = FastAPI(title="VitaMind Spark agent", version=SPARK_VERSION, lifespan=lifespan,
              description="ADHD executive-function assistant for the Lumina ecosystem. "
                          "Local models only; ADHD track only; signed callers only.")


@app.middleware("http")
async def _authenticate(request: Request, call_next):
    if request.url.path in _auth.PUBLIC_PATHS:
        return await call_next(request)

    body = await request.body()
    request_id = request.headers.get(_auth.REQUEST_ID_HEADER) or str(uuid.uuid4())

    if _secret is not None:
        try:
            request_id = _auth.verify(_secret, request.headers, body)
        except _auth.AuthError as error:
            return JSONResponse(status_code=401,
                                content={"error": "unauthorized", "reason": error.reason},
                                headers={_auth.REQUEST_ID_HEADER: request_id})

    request.state.request_id = request_id
    started = perf_counter()
    response: Response = await call_next(request)
    response.headers[_auth.REQUEST_ID_HEADER] = request_id
    response.headers["x-processing-ms"] = f"{(perf_counter() - started) * 1000:.2f}"
    return response


def _require_assistant() -> LuminaADHD:
    if _assistant is None:
        raise RuntimeError(_startup_error or "assistant not loaded")
    return _assistant


def _model_summary(assistant: LuminaADHD) -> dict:
    intent, friction = assistant.intent_classifier.model, assistant.friction_classifier.model
    journal = getattr(assistant.safety.agent, "model", None)
    return {
        "intent_classifier": {
            "loaded": intent.artifact is not None,
            "version": intent.metadata.get("modelVersion", "rules"),
            "role": "routes the message; explicit-command guard and rules fall back"},
        "friction_classifier": {
            "loaded": friction.artifact is not None,
            "version": friction.metadata.get("modelVersion", "rules"),
            "role": "names the block that selects the struggle-support template"},
        "safety_journal_model": {
            "loaded": journal is not None,
            "version": getattr(journal, "version", None),
            "role": "Lumina's journal_ai (shared copy): runs before any planning"},
    }


# --- ops ----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE, "version": SPARK_VERSION,
            "clinically_validated": False}


@app.get("/ready")
def ready():
    assistant = _assistant
    if assistant is None:
        return JSONResponse(status_code=503, content={
            "status": "not_ready", "service": SERVICE, "reason": _startup_error})
    if not assistant.ready:
        return JSONResponse(status_code=503, content={
            "status": "not_ready", "service": SERVICE,
            "reason": "safety model unavailable"})
    return {"status": "ready", "service": SERVICE, "track": SPARK_TRACK,
            "signed_requests_required": _secret is not None,
            "models": _model_summary(assistant)}


@app.get("/version")
def version():
    return {"service": SERVICE, "agent_version": SPARK_VERSION,
            "contract_version": CONTRACT_VERSION, "track": SPARK_TRACK,
            "no_external_inference_api": True, "no_pretrained_weights": True}


# --- the agent ----------------------------------------------------------
def _error(status, error, reason, **extra):
    return JSONResponse(status_code=status, content={"error": error, "reason": reason, **extra})


@app.post(API + "/organize")
def organize(request: Request, payload: dict = Body(...)):
    """One Spark turn. The body is parsed by hand so the track is judged first:
    a non-ADHD caller gets 403 whatever else is wrong with the payload."""
    request_id = getattr(request.state, "request_id", None)
    track = payload.get("track")
    if track is None:
        return _error(422, "invalid_request", "track is required")
    if track != SPARK_TRACK:
        logger.warning("requestId=%s refused: track=%s", request_id,
                       track if isinstance(track, str) and len(track) <= 20 else "invalid")
        return _error(403, "spark_adhd_only",
                      "Spark is only available to patients on the ADHD track")
    try:
        bundle = SparkOrganizeRequest.model_validate(payload)
    except ValidationError as error:
        # Locations and types only: an error message can echo patient text.
        return _error(422, "invalid_request", "request does not match the contract",
                      fields=[{"location": [str(part) for part in e["loc"]], "type": e["type"]}
                              for e in error.errors()])
    if bundle.contract_version != CONTRACT_VERSION:
        return _error(409, "contract_mismatch",
                      f"expected {CONTRACT_VERSION}, received {bundle.contract_version}")
    if bundle.request.calendar.commit:
        return _error(422, "invalid_request",
                      "calendar.commit is not supported: the backend persists tasks")

    assistant = _require_assistant()
    began, status, code = perf_counter(), "OK", None
    try:
        result = assistant.organize(bundle.request)
        status = ("SAFETY_HANDOFF_REQUIRED" if result.safety.level != "NORMAL"
                  else "CLARIFICATION" if result.analysis.needsClarification else "OK")
    except AssistantError as error:
        status, code = "ERROR", error.code
        return _error(error.status, error.code.lower(), error.message)
    except Exception:
        status, code = "ERROR", "INTERNAL_ERROR"
        # Never include exception text: it can contain patient text.
        return _error(500, "internal_error", "the assistant could not complete this request")
    finally:
        logger.info("requestId=%s module=SPARK operation=organize language=%s "
                    "latencyMs=%.2f resultStatus=%s errorCode=%s", request_id,
                    bundle.request.patient.language, (perf_counter() - began) * 1000,
                    status, code)

    persistence = ["PERSIST_INTERACTION"]
    if result.taskOperations:
        persistence.append("PERSIST_TASK_OPERATIONS")
    if result.safety.level != "NORMAL":
        persistence.append("PERSIST_SAFETY_EVENT")
    return {
        "requestId": str(bundle.request.requestId),
        "agent": AGENT,
        "agentVersion": SPARK_VERSION,
        "contractVersion": CONTRACT_VERSION,
        "track": SPARK_TRACK,
        "result": result.model_dump(mode="json"),
        "persistence": persistence,
        "clinicallyValidated": False,
    }


@app.exception_handler(RuntimeError)
def _assistant_unavailable(request: Request, error: RuntimeError):
    # A missing assistant is a service fault and must never look like a safe reply.
    return JSONResponse(status_code=503, content={
        "error": "assistant_unavailable", "reason": str(error)})


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("SPARK_HOST", "127.0.0.1"),
                port=int(os.environ.get("SPARK_PORT", "8103")))


if __name__ == "__main__":
    main()
