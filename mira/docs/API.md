# Optional local API

Start from the project root:

```sh
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the local interactive API documentation.

1. `POST /api/v1/mira/session` with `{"language":"en"}` or `{"language":"ar"}`.
2. Keep the response's `session_id` and `session_token`.
3. For message, read and delete requests, send the header
   `X-Mira-Session-Token: <session_token>`.
4. `POST /api/v1/mira/session/<session_id>/message` with
   `{"text":"your answer","language":"en"}`. Use the session's language.
5. `GET /api/v1/mira/session/<session_id>` reads the current session/report.
6. `DELETE /api/v1/mira/session/<session_id>` deletes that process-local session.

This token is generated locally. It is not an AI-provider API key.
Invalid request shapes return HTTP 422; a language mismatch returns 400; missing
or invalid session tokens return 403 for an existing session.

Send `report`, `summary`, `تقرير`, or `ملخص` as a message to request a report.
Reports include `report_ready`, `assessment_complete` and `result.report_text`.
`assessment_complete: true` does not block subsequent corrections or safety
support. Urgent replies have `assessment_complete: false`, `input_enabled: true`
and `safety.notification_sent: false`.

Sessions disappear when the process restarts. This is a local development API,
not a deployed patient service; a capability token is not a complete account,
consent or authorization system.
