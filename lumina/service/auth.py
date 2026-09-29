"""Service-to-service authentication for the agent boundary.

Only the backend may call Lumina. The scheme is HMAC-SHA256 over
`timestamp.requestId.sha256(body)`, which binds a signature to one specific body
at one specific moment - so a captured request cannot be replayed later or
repointed at a different payload.

Health, readiness and version are deliberately outside the scheme: an orchestrator
has to be able to probe a service without holding its secret.

If no secret is configured the service refuses to start in production mode rather
than quietly accepting unsigned requests, which is the failure that turns an
internal agent into an open endpoint.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

SIGNATURE_HEADER = "x-agent-signature"
TIMESTAMP_HEADER = "x-agent-timestamp"
REQUEST_ID_HEADER = "x-request-id"

# A window wide enough for clock skew, narrow enough that a captured signature is
# useless within a minute.
MAX_CLOCK_SKEW_SECONDS = 60

PUBLIC_PATHS = ("/health", "/ready", "/version", "/openapi.json", "/docs",
                "/docs/oauth2-redirect", "/redoc")


class AuthError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def body_digest(body: bytes) -> str:
    return hashlib.sha256(body or b"").hexdigest()


def sign(secret: str, timestamp: str, request_id: str, body: bytes) -> str:
    """Produce the expected signature. Shared with the backend client by contract."""
    message = f"{timestamp}.{request_id}.{body_digest(body)}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify(secret: str, headers, body: bytes, now=None) -> str:
    """Verify a signed request and return its request id.

    Raises AuthError with a reason that is safe to log - it never echoes the
    signature or the secret.
    """
    signature = headers.get(SIGNATURE_HEADER)
    timestamp = headers.get(TIMESTAMP_HEADER)
    request_id = headers.get(REQUEST_ID_HEADER)

    if not signature or not timestamp or not request_id:
        raise AuthError("missing_signature_headers")

    try:
        sent_at = float(timestamp)
    except (TypeError, ValueError):
        raise AuthError("malformed_timestamp")

    now = now if now is not None else time.time()
    if abs(now - sent_at) > MAX_CLOCK_SKEW_SECONDS:
        raise AuthError("timestamp_outside_window")

    expected = sign(secret, timestamp, request_id, body)
    if not hmac.compare_digest(expected, signature):
        raise AuthError("signature_mismatch")

    return request_id


def load_secret(env="LUMINA_SERVICE_SECRET", allow_missing=None):
    """Read the shared secret, refusing to run unauthenticated in production.

    `allow_missing` defaults to the LUMINA_ALLOW_UNSIGNED flag so a developer can
    run the service without a secret, and a deployment cannot do so by accident.
    """
    secret = os.environ.get(env, "").strip()
    if secret:
        return secret
    if allow_missing is None:
        allow_missing = os.environ.get("LUMINA_ALLOW_UNSIGNED", "false").lower() == "true"
    if allow_missing:
        return None
    raise RuntimeError(
        f"{env} is not set. Set it, or set LUMINA_ALLOW_UNSIGNED=true for local "
        "development only - never in production.")
