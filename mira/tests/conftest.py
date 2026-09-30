"""Make the test clients behave like the real one: hold and send the session token.

Mira issues a per-session capability token at session start and refuses every later
call without `X-Mira-Session-Token` (403). The API tests were written before that
requirement and call the message endpoint bare, so they failed on the missing header
rather than on anything they assert. Real clients (the VitaMind backend) remember the
token from the start response and send it; this does the same for `TestClient`.

It adds a header only, and only when the test did not send one itself. A test module
that deals with the token itself (it mentions the header) is left completely alone, so
the tests that deliberately omit or forge it still exercise the 403 path.
"""
import inspect
import re

import pytest
from fastapi.testclient import TestClient

SESSION_PATH = re.compile(r"^/api/v1/mira/session/([^/?]+)")


@pytest.fixture(autouse=True)
def _remember_session_tokens(request, monkeypatch):
    if "X-Mira-Session-Token" in inspect.getsource(request.module):
        yield
        return
    tokens = {}
    original = TestClient.request

    def with_token(self, method, url, *args, **kwargs):
        match = SESSION_PATH.match(str(url))
        headers = dict(kwargs.get("headers") or {})
        if match and match.group(1) in tokens and "x-mira-session-token" not in {
                key.lower() for key in headers}:
            headers["X-Mira-Session-Token"] = tokens[match.group(1)]
            kwargs["headers"] = headers
        response = original(self, method, url, *args, **kwargs)
        if str(url).rstrip("/") == "/api/v1/mira/session" and method.upper() == "POST":
            try:
                body = response.json()
            except ValueError:
                body = {}
            if body.get("session_id") and body.get("session_token"):
                tokens[body["session_id"]] = body["session_token"]
        return response

    monkeypatch.setattr(TestClient, "request", with_token)
    yield
