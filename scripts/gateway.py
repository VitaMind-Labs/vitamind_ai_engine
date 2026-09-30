"""Single public entry point in front of the three agents.

Why this exists: a Render web service receives traffic on exactly one port
(`$PORT`), and free instances cannot be reached over the private network at all.
Mira (8101), Lumina (8102) and Spark (8103) each keep their own port *inside* the
instance, so they still fail and restart independently; this gateway is the one
port the backend can reach, and it routes by path prefix:

    https://<service>/mira/...    -> 127.0.0.1:8101/...
    https://<service>/lumina/...  -> 127.0.0.1:8102/...
    https://<service>/spark/...   -> 127.0.0.1:8103/...

It is a dumb pipe. Requests are forwarded byte for byte - method, query, body and
every header, including the HMAC signature headers - so signature verification
still happens in the agent and the gateway holds no secret. Stdlib only.
"""
from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# RFC 7230 s6.1: describe one connection, never forwarded.
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}

MAX_BODY_BYTES = 2 * 1024 * 1024  # larger than any agent contract allows
UPSTREAM_TIMEOUT_SECONDS = 30


def split_route(path: str, prefixes):
    """('/lumina/api/v1/x?y=1') -> ('lumina', '/api/v1/x?y=1'); None if no match."""
    for name in prefixes:
        prefix = "/" + name
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            rest = path[len(prefix):] or "/"
            return name, rest if rest.startswith("/") else "/" + rest
    return None


def make_gateway(host: str, port: int, upstreams: dict, status=None):
    """`upstreams` maps a prefix to its local port; `status()` reports agent health."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "vitamind-gateway"

        def log_message(self, fmt, *args):  # path only, never headers or bodies
            print(f"[gateway] {self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''}",
                  flush=True)

        def _reply(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _own_routes(self) -> bool:
            path = self.path.split("?")[0]
            if path in ("/", "/health"):
                # Liveness of the gateway itself. It must stay 200 while an agent
                # restarts, or the platform would recycle all three for one crash.
                self._reply(200, {"status": "ok", "service": "vitamind-agents",
                                  "agents": status() if status else {}})
                return True
            if path == "/ready":
                agents = status() if status else {}
                ok = bool(agents) and all(a.get("ready") for a in agents.values())
                self._reply(200 if ok else 503, {"status": "ready" if ok else "not_ready",
                                                 "agents": agents})
                return True
            return False

        def _proxy(self):
            if self._own_routes():
                return
            route = split_route(self.path, upstreams)
            if route is None:
                self._reply(404, {"error": "unknown_route",
                                  "routes": ["/" + name for name in upstreams]})
                return
            name, rest = route

            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._reply(413, {"error": "payload_too_large"})
                return
            body = self.rfile.read(length) if length else None

            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            if body is not None:
                headers["Content-Length"] = str(len(body))
            try:
                connection = http.client.HTTPConnection(
                    "127.0.0.1", upstreams[name], timeout=UPSTREAM_TIMEOUT_SECONDS)
                connection.request(self.command, rest, body=body, headers=headers)
                upstream = connection.getresponse()
                payload = upstream.read()
            except TimeoutError:
                self._reply(504, {"error": "agent_timeout", "agent": name})
                return
            except OSError:
                self._reply(502, {"error": "agent_unreachable", "agent": name})
                return
            finally:
                try:
                    connection.close()
                except Exception:
                    pass

            self.send_response(upstream.status)
            for key, value in upstream.getheaders():
                if key.lower() not in HOP_BY_HOP:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _proxy

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve_in_thread(server) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, name="gateway", daemon=True)
    thread.start()
    return thread
