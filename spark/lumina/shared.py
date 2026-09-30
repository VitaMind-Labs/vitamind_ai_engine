"""Components Spark takes from the Lumina agent, so there is one copy of each.

Spark is Lumina's ADHD specialist, so two things must not be forked:

* `journal_ai` - the safety rules and journal classifier. Spark shipped its own
  byte-identical copy (same 11 modules, same weights sha256), which is a second
  place for the safety model to drift. It now loads Lumina's.
* the service-to-service HMAC scheme (`lumina/service/auth.py`). The backend
  signs every agent call the same way, so Spark verifies with the same code
  rather than a re-implementation that could diverge.

This package is *also* called `lumina` (it was written as `lumina.adhd`), so
Lumina's `lumina/` directory must never be put ahead of Spark's on `sys.path`:
Spark's own modules would be shadowed. Lumina's directory is only ever appended,
`journal_ai` has a unique top-level name, and Lumina's auth module is loaded by
file path under a private name. The two agents stay separate processes.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

SPARK_ROOT = Path(__file__).resolve().parents[1]
# vitamind_agent/spark/lumina/shared.py -> vitamind_agent/lumina_agent
# ("lumina" is the pre-rename folder name, still accepted for older checkouts).
def _default_lumina_root() -> Path:
    for name in ("lumina_agent", "lumina"):
        candidate = SPARK_ROOT.parent / name
        if (candidate / "journal_ai").is_dir():
            return candidate
    return SPARK_ROOT.parent / "lumina_agent"


LUMINA_ROOT = Path(os.environ.get("LUMINA_ROOT") or _default_lumina_root()).resolve()

_auth_module = None


class SharedComponentError(RuntimeError):
    pass


def _require_lumina_root():
    if not (LUMINA_ROOT / "journal_ai").is_dir():
        raise SharedComponentError(
            f"Lumina components not found at {LUMINA_ROOT}. Spark shares journal_ai "
            "and the service auth scheme with the Lumina agent; deploy the two "
            "folders side by side, or set LUMINA_ROOT.")


def journal_ai():
    """Lumina's `journal_ai` package (safety rules + journal classifier)."""
    _require_lumina_root()
    root = str(LUMINA_ROOT)
    if root not in sys.path:
        sys.path.append(root)  # append, never insert: see the module docstring
    module = importlib.import_module("journal_ai")
    # Guard the one failure this arrangement can produce: `lumina` resolving to
    # Lumina's package instead of Spark's.
    own = sys.modules.get("lumina")
    if own is not None and getattr(own, "__file__", None):
        if SPARK_ROOT not in Path(own.__file__).resolve().parents:
            raise SharedComponentError(
                "`lumina` resolves to the Lumina agent's package inside the Spark "
                "process; run Spark from its own directory.")
    return module


def service_auth():
    """Lumina's HMAC helpers (`sign`, `verify`, `AuthError`, PUBLIC_PATHS...)."""
    global _auth_module
    if _auth_module is None:
        _require_lumina_root()
        path = LUMINA_ROOT / "service" / "auth.py"
        spec = importlib.util.spec_from_file_location("lumina_service_auth", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _auth_module = module
    return _auth_module
