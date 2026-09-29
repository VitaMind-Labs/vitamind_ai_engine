"""Start Mira and Lumina together, each in its own process on its own port.

Why separate OS processes rather than one ASGI app mounting both: the two agents
must fail independently. Mounted together, an unhandled exception during startup,
a blocking model load, or a memory spike in one takes the other down with it — and
the whole point of the split is that a Mira outage must not stop Lumina support
for patients who are already onboarded.

    python -m scripts.run_all                 # both, with restart-on-crash
    python -m scripts.run_all --only lumina   # one
    python -m scripts.run_all --reload        # dev autoreload
    python -m scripts.run_all --no-restart    # fail fast instead of respawning

Stdlib only: subprocess, threading, signal. No orchestrator required to run the
stack on a laptop.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ANSI colours, disabled when not writing to a terminal so log files stay clean.
_TTY = sys.stdout.isatty()


def paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


@dataclass
class ServiceSpec:
    name: str
    colour: str
    cwd: Path
    app: str
    port_env: str
    default_port: int
    # Environment variables this service cannot start without.
    required_env: tuple = ()
    # Extra defaults injected when the operator has not set them.
    env_defaults: dict = field(default_factory=dict)


SERVICES = {
    "mira": ServiceSpec(
        name="mira",
        colour="36",  # cyan
        cwd=ROOT / "mira",
        app="app:app",
        port_env="MIRA_PORT",
        default_port=8101,
    ),
    "lumina": ServiceSpec(
        name="lumina",
        colour="35",  # magenta
        cwd=ROOT / "lumina",
        app="service.app:app",
        port_env="LUMINA_PORT",
        default_port=8102,
        # Lumina signs service-to-service calls. Running it unsigned is a local
        # development choice that must be made explicitly, never by omission.
        required_env=("LUMINA_SERVICE_SECRET",),
        env_defaults={"ALLOW_UNREVIEWED_SAFETY_CONTENT": "true"},
    ),
}

RESTART_BACKOFF = (1, 2, 5, 10, 30)
READY_TIMEOUT_SECONDS = 90


class ProcessGuard:
    """Make the children die with the launcher, even if it is killed outright.

    The graceful path (Ctrl+C / SIGTERM) is handled by the signal handler below,
    but that handler cannot run if the launcher is SIGKILLed — and then both agents
    survive holding their ports, so the *next* run fails with "address in use".
    That was observed, not theorised.

    On Windows the children are put in a Job Object with KILL_ON_JOB_CLOSE: when
    the launcher's last handle goes away, the OS terminates the whole job. On
    POSIX the children already get their own session via `start_new_session`, and
    `prctl(PR_SET_PDEATHSIG)` is applied in the child so an orphan self-terminates.
    """

    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self.handle = kernel32.CreateJobObjectW(None, None)
            if not self.handle:
                return

            # JOBOBJECT_EXTENDED_LIMIT_INFORMATION, only the fields we need.
            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in
                            ("ReadOperationCount", "WriteOperationCount",
                             "OtherOperationCount", "ReadTransferCount",
                             "WriteTransferCount", "OtherTransferCount")]

            class BASIC_LIMIT(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class EXTENDED_LIMIT(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", BASIC_LIMIT),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
            JobObjectExtendedLimitInformation = 9

            info = EXTENDED_LIMIT()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            kernel32.SetInformationJobObject(
                self.handle, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info))
            self._kernel32 = kernel32
        except Exception:
            # A missing job object is a degraded guarantee, not a reason not to
            # run. The graceful shutdown path still works.
            self.handle = None

    def adopt(self, process: subprocess.Popen):
        if self.handle is None or os.name != "nt":
            return
        try:
            import ctypes

            PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
            child = self._kernel32.OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, process.pid)
            if child:
                self._kernel32.AssignProcessToJobObject(self.handle, child)
                self._kernel32.CloseHandle(child)
        except Exception:
            pass

    @staticmethod
    def preexec():
        """POSIX: ask the kernel to SIGTERM this child if the launcher dies."""
        if os.name == "nt":
            return None

        def _set_pdeathsig():
            try:
                import ctypes

                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                PR_SET_PDEATHSIG = 1
                libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
            except Exception:
                pass

        return _set_pdeathsig


def load_dotenv(path: Path) -> dict:
    """Minimal .env reader. Values are never printed."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def probe(url: str, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read().decode("utf-8"))
        except Exception:
            return error.code, None
    except Exception:
        return None, None


class Runner:
    """Owns one service process: spawn, log-pump, restart, terminate."""

    def __init__(self, spec: ServiceSpec, environment: dict, reload_: bool,
                 restart: bool, guard: "ProcessGuard | None" = None):
        self.spec = spec
        self.environment = environment
        self.reload = reload_
        self.restart = restart
        self.guard = guard
        self.port = int(environment.get(spec.port_env, spec.default_port))
        self.process: subprocess.Popen | None = None
        self.restarts = 0
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def log(self, message: str, stream=None):
        prefix = paint(self.spec.colour, f"[{self.spec.name}]")
        print(f"{prefix} {message}", flush=True, file=stream or sys.stdout)

    def _command(self) -> list[str]:
        command = [sys.executable, "-m", "uvicorn", self.spec.app,
                   "--host", self.environment.get("AGENT_HOST", "127.0.0.1"),
                   "--port", str(self.port), "--log-level",
                   self.environment.get("AGENT_LOG_LEVEL", "info")]
        if self.reload:
            command.append("--reload")
        return command

    def _spawn(self) -> subprocess.Popen:
        env = dict(os.environ)
        env.update(self.environment)
        env.update(self.spec.env_defaults)
        env.setdefault(self.spec.port_env, str(self.port))
        # Unbuffered so a child's logs interleave in real time rather than
        # appearing in blocks after it exits.
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            self._command(),
            cwd=str(self.spec.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # A new process group means Ctrl+C reaches the parent first, so it can
            # shut children down in order instead of them dying mid-write.
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
            preexec_fn=ProcessGuard.preexec() if os.name != "nt" else None,
        )
        # Adopted into the job object so an abruptly killed launcher cannot leave
        # this agent holding its port.
        if self.guard is not None:
            self.guard.adopt(process)
        return process

    def _pump(self):
        """Prefix and forward child output until the process ends."""
        while not self.stopping.is_set():
            self.process = self._spawn()
            self.log(f"started pid={self.process.pid} port={self.port}")
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.log(line.rstrip())
            code = self.process.wait()

            if self.stopping.is_set():
                return
            if not self.restart:
                self.log(paint("31", f"exited with code {code}; restart disabled"))
                return

            backoff = RESTART_BACKOFF[min(self.restarts, len(RESTART_BACKOFF) - 1)]
            self.restarts += 1
            # Only this service restarts. The other keeps serving.
            self.log(paint("33", f"exited with code {code}; restarting in {backoff}s "
                                 f"(restart #{self.restarts})"))
            if self.stopping.wait(backoff):
                return

    def start(self):
        self.thread = threading.Thread(target=self._pump, name=f"{self.spec.name}-runner",
                                       daemon=True)
        self.thread.start()

    def wait_ready(self, deadline: float):
        """Poll /health then /ready. Returns (health_ok, ready_ok, version)."""
        health_ok = False
        while time.time() < deadline:
            if self.stopping.is_set():
                return False, False, None
            if not health_ok:
                status, _ = probe(f"{self.base_url}/health")
                health_ok = status == 200
            if health_ok:
                status, body = probe(f"{self.base_url}/ready")
                if status == 200:
                    _, version = probe(f"{self.base_url}/version")
                    return True, True, version
                # Healthy but not ready is a real state worth waiting through:
                # the process is up and still loading its models.
            time.sleep(0.5)
        _, version = probe(f"{self.base_url}/version")
        return health_ok, False, version

    def stop(self, timeout=10.0):
        self.stopping.set()
        process = self.process
        if process is None or process.poll() is not None:
            return
        self.log("stopping")
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            process.wait(timeout=timeout)
        except Exception:
            self.log(paint("31", "did not stop in time; killing"))
            try:
                process.kill()
            except Exception:
                pass


def status_table(runners: list[Runner], results: dict) -> str:
    header = f"{'SERVICE':<10} {'URL':<26} {'HEALTH':<9} {'READY':<9} VERSION"
    lines = [header, "-" * len(header)]
    for runner in runners:
        health, ready, version = results.get(runner.spec.name, (False, False, None))
        agent_version = (version or {}).get("agent_version") or "-"
        contract = (version or {}).get("contract_version") or "-"
        lines.append(
            f"{runner.spec.name:<10} {runner.base_url:<26} "
            f"{paint('32', 'ok') if health else paint('31', 'DOWN'):<9} "
            f"{paint('32', 'ready') if ready else paint('33', 'not-ready'):<9} "
            f"{agent_version} ({contract})"
        )
    return "\n".join(lines)


def validate_environment(specs, environment) -> list[str]:
    """Fail fast on missing configuration, without ever echoing a value."""
    problems = []
    for spec in specs:
        for key in spec.required_env:
            if not environment.get(key) and not os.environ.get(key):
                problems.append(
                    f"{spec.name}: {key} is not set. Set it in .env, or export "
                    f"LUMINA_ALLOW_UNSIGNED=true for local development only.")
        if not spec.cwd.exists():
            problems.append(f"{spec.name}: directory not found: {spec.cwd}")
    ports = {}
    for spec in specs:
        port = int(environment.get(spec.port_env, spec.default_port))
        if port in ports:
            problems.append(
                f"port {port} requested by both {ports[port]} and {spec.name}; "
                "each agent needs its own port")
        ports[port] = spec.name
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(SERVICES), action="append",
                        help="run just this service (repeatable)")
    parser.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev)")
    parser.add_argument("--no-restart", action="store_true",
                        help="do not respawn a crashed service")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT_SECONDS)
    args = parser.parse_args()

    environment = load_dotenv(Path(args.env_file))
    # LUMINA_ALLOW_UNSIGNED is the documented escape hatch; honour it here too so
    # the launcher's own validation agrees with the service's.
    # Compare the value, not its presence: .env.example ships "false".
    if (environment.get("LUMINA_ALLOW_UNSIGNED", "").lower() == "true"
            or os.environ.get("LUMINA_ALLOW_UNSIGNED", "").lower() == "true"):
        SERVICES["lumina"].required_env = ()

    selected = [SERVICES[name] for name in (args.only or sorted(SERVICES))]

    problems = validate_environment(selected, environment)
    if problems:
        print(paint("31", "Configuration problems:"), file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    guard = ProcessGuard()
    runners = [Runner(spec, environment, args.reload, not args.no_restart, guard)
               for spec in selected]

    stopping = threading.Event()

    def shutdown(_signum=None, _frame=None):
        if stopping.is_set():
            return
        stopping.set()
        print()
        print(paint("33", "shutting down both agents..."))
        for runner in runners:
            runner.stop()

    signal.signal(signal.SIGINT, shutdown)
    try:
        signal.signal(signal.SIGTERM, shutdown)
    except (AttributeError, ValueError):
        pass  # not available on every platform

    for runner in runners:
        runner.start()

    deadline = time.time() + args.ready_timeout
    results = {}
    for runner in runners:
        results[runner.spec.name] = runner.wait_ready(deadline)

    print()
    print(status_table(runners, results))
    print()
    down = [name for name, (health, _, _) in results.items() if not health]
    if down:
        print(paint("31", f"not reachable: {', '.join(down)}"))
    not_ready = [name for name, (health, ready, _) in results.items() if health and not ready]
    if not_ready:
        print(paint("33", f"reachable but not ready: {', '.join(not_ready)} "
                          "(check /ready for the reason)"))
    if not down and not not_ready:
        print(paint("32", "both agents up. Ctrl+C to stop."))

    try:
        while not stopping.is_set():
            # One service dying never ends the loop: its own runner restarts it,
            # and the other keeps serving throughout.
            alive = [r for r in runners if r.thread and r.thread.is_alive()]
            if not alive:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()

    for runner in runners:
        if runner.thread:
            runner.thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
