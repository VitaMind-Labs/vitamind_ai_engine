"""The single-file deployment: launcher validation and the public gateway."""
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scripts import run_all
from scripts.gateway import make_gateway, serve_in_thread, split_route


# --- routing ------------------------------------------------------------
@pytest.mark.parametrize("path,expected", [
    ("/lumina/api/v1/lumina/chat", ("lumina", "/api/v1/lumina/chat")),
    ("/spark/health", ("spark", "/health")),
    ("/mira", ("mira", "/")),
    ("/mira?x=1", ("mira", "/?x=1")),
    ("/lumina/x?a=1&b=2", ("lumina", "/x?a=1&b=2")),
    ("/luminae/x", None),          # a prefix is a whole path segment, not a substring
    ("/sparkle", None),
    ("/", None),
    ("/api/v1/mira/session", None),
])
def test_split_route(path, expected):
    assert split_route(path, ("mira", "lumina", "spark")) == expected


# --- gateway against fake agents ---------------------------------------
class Echo(BaseHTTPRequestHandler):
    """Reports exactly what reached the agent."""
    label = "?"

    def _go(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode()
        payload = json.dumps({"agent": self.label, "method": self.command, "path": self.path,
                              "body": body,
                              "headers": {k.lower(): v for k, v in self.headers.items()}}).encode()
        code = 403 if self.path.startswith("/forbidden") else 200
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-request-id", self.headers.get("x-request-id", ""))
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_DELETE = _go

    def log_message(self, *args):
        pass


@pytest.fixture()
def stack():
    servers = []
    ports = {}
    for name in ("mira", "lumina", "spark"):
        handler = type(f"Echo_{name}", (Echo,), {"label": name})
        server = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        ports[name] = server.server_address[1]
    gateway = make_gateway("127.0.0.1", 0, ports,
                           lambda: {n: {"up": True, "ready": True} for n in ports})
    serve_in_thread(gateway)
    yield gateway.server_address[1], ports
    gateway.shutdown()
    for server in servers:
        server.shutdown()


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, dict(response.getheaders()), data


def test_each_prefix_reaches_its_own_agent(stack):
    port, _ = stack
    for name in ("mira", "lumina", "spark"):
        status, _, data = request(port, "GET", f"/{name}/health")
        assert status == 200 and json.loads(data)["agent"] == name
        assert json.loads(data)["path"] == "/health"


def test_signature_headers_body_and_status_pass_through_untouched(stack):
    port, _ = stack
    body = json.dumps({"track": "ADHD"}, separators=(",", ":"))
    headers = {"Content-Type": "application/json", "x-agent-signature": "abc123",
               "x-agent-timestamp": "1700000000.000", "x-request-id": "req-9"}
    status, response_headers, data = request(port, "POST", "/spark/api/v1/spark/organize?x=1",
                                             body, headers)
    seen = json.loads(data)
    assert status == 200
    assert seen["body"] == body and seen["path"] == "/api/v1/spark/organize?x=1"
    for key, value in headers.items():
        assert seen["headers"][key.lower()] == value
    assert response_headers["x-request-id"] == "req-9"


def test_agent_error_statuses_are_not_rewritten(stack):
    port, _ = stack
    status, _, _ = request(port, "POST", "/spark/forbidden", "{}",
                           {"Content-Type": "application/json"})
    assert status == 403


def test_unknown_route_is_404_and_not_forwarded(stack):
    port, _ = stack
    status, _, data = request(port, "GET", "/admin/secrets")
    assert status == 404 and "agent" not in json.loads(data)


def test_dead_agent_is_a_502_and_the_others_keep_serving(stack):
    _, ports = stack
    gateway = make_gateway("127.0.0.1", 0, {**ports, "spark": 1}, lambda: {})
    serve_in_thread(gateway)
    try:
        gport = gateway.server_address[1]
        assert request(gport, "GET", "/spark/health")[0] == 502
        assert request(gport, "GET", "/lumina/health")[0] == 200
    finally:
        gateway.shutdown()


def test_gateway_health_stays_200_while_an_agent_is_down():
    gateway = make_gateway("127.0.0.1", 0, {"mira": 1},
                           lambda: {"mira": {"up": False, "ready": False}})
    serve_in_thread(gateway)
    try:
        port = gateway.server_address[1]
        assert request(port, "GET", "/health")[0] == 200
        assert request(port, "GET", "/ready")[0] == 503
    finally:
        gateway.shutdown()


# --- launcher -----------------------------------------------------------
def test_three_agents_have_distinct_default_ports():
    ports = {name: spec.default_port for name, spec in run_all.SERVICES.items()}
    assert ports == {"mira": 8101, "lumina": 8102, "spark": 8103}


def test_spark_needs_a_secret_from_either_name(monkeypatch):
    for name in ("SPARK_SERVICE_SECRET", "LUMINA_SERVICE_SECRET"):
        monkeypatch.delenv(name, raising=False)
    spark = run_all.SERVICES["spark"]
    problems = run_all.validate_environment([spark], {})
    assert any("SPARK_SERVICE_SECRET" in p for p in problems)
    assert not run_all.validate_environment([spark], {"LUMINA_SERVICE_SECRET": "x" * 32})
    assert not run_all.validate_environment([spark], {"SPARK_SERVICE_SECRET": "x" * 32})


def test_gateway_port_may_not_collide_with_an_agent():
    specs = list(run_all.SERVICES.values())
    env = {"LUMINA_SERVICE_SECRET": "x" * 32}
    assert not run_all.validate_environment(specs, env, gateway_port=10000)
    assert any("gateway" in p for p in run_all.validate_environment(specs, env, gateway_port=8102))


def test_two_agents_cannot_share_a_port():
    specs = list(run_all.SERVICES.values())
    env = {"LUMINA_SERVICE_SECRET": "x" * 32, "SPARK_PORT": "8101"}
    assert any("8101" in p for p in run_all.validate_environment(specs, env))


def test_real_environment_variables_override_the_env_file(monkeypatch, tmp_path):
    """Render has no .env: ports and secrets arrive in the process environment."""
    monkeypatch.setenv("MIRA_PORT", "9101")
    monkeypatch.setenv("LUMINA_SERVICE_SECRET", "s" * 32)
    env_file = tmp_path / ".env"
    env_file.write_text("MIRA_PORT=8101\nLUMINA_PORT=8102\n", encoding="utf-8")
    captured = {}

    class Stop(Exception):
        pass

    def fake_validate(specs, environment, gateway_port=None):
        captured.update(environment)
        raise Stop

    monkeypatch.setattr(run_all, "validate_environment", fake_validate)
    monkeypatch.setattr("sys.argv", ["run.py", "--env-file", str(env_file), "--no-gateway"])
    with pytest.raises(Stop):
        run_all.main()
    assert captured["MIRA_PORT"] == "9101"      # environment beats the file
    assert captured["LUMINA_PORT"] == "8102"    # the file still supplies the rest
