"""FastAPI HTTP adapter for the framework-independent Mira application service."""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import Literal

from pydantic import BaseModel, Field

from vitamind.mira.service import MiraSessionService
from vitamind.mira.interview import DEFAULT_SESSION_QUESTIONS

app = FastAPI(title="VitaMind Mira AI Service", version="1.0.0")
sessions = MiraSessionService()


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "invalid request: language must be 'en' or 'ar'"})


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    language: Literal["en", "ar"]


class StartSessionRequest(BaseModel):
    language: Literal["en", "ar"]


def reply_payload(session, reply):
    payload = asdict(reply)
    payload["session_id"] = session.session_id
    payload["assistant_message"] = payload.pop("text")
    payload["language"] = session.language
    return payload


@app.get("/health")
def health():
    return {"status": "ok", "service": "mira", "version": app.version}


@app.post("/api/v1/mira/session")
def start_session(request: StartSessionRequest | None = None):
    if request is None:
        raise HTTPException(status_code=400, detail="language must be 'en' or 'ar'")
    try:
        session, reply = sessions.start(request.language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "session_id": session.session_id,
        "assistant_message": reply.text,
        "chapter": reply.chapter,
        "chapter_progress": reply.chapter_progress,
        "assessment_complete": False,
        "safety": {"level": "routine", "flags": []},
        "result": None,
        "language": session.language,
    }


@app.post("/api/v1/mira/session/{session_id}/message")
def send_message(session_id: str, request: MessageRequest):
    try:
        session = sessions.get(session_id)
        if request.language != session.language:
            raise HTTPException(status_code=400, detail="language does not match session")
        session, reply = sessions.message(session_id, request.text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return reply_payload(session, reply)


@app.get("/api/v1/mira/session/{session_id}")
def get_session(session_id: str):
    try:
        session = sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    safety = session.state.safety_snapshot()
    return {
        "session_id": session.session_id,
        "chapter": session.chapter,
        "chapter_progress": min(1.0, session.messages / DEFAULT_SESSION_QUESTIONS),
        "complete": session.complete,
        "assessment_complete": session.complete,
        "assistant_message": session.assistant_message,
        "safety": {"level": "urgent" if safety["urgent"] else "routine", "flags": [item["feature"] for item in safety["active_features"]]},
        "result": session.result,
        "language": session.language,
    }


@app.delete("/api/v1/mira/session/{session_id}")
def delete_session(session_id: str):
    sessions.delete(session_id)
    return {"success": True}