"""Write or verify Spark's MANIFEST.json (SHA-256 of every shipped file).

    python -m training.build_manifest           # regenerate MANIFEST.json
    python -m training.build_manifest --check   # exit 1 if any file drifted

Text files are hashed with LF line endings so the manifest is the same on a Windows
checkout (CRLF) and on Linux/Render; binaries are hashed as they are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", "runtime_data", ".git"}
SKIP_SUFFIXES = (".pyc", ".sqlite3", ".sqlite3-journal")
BINARY_SUFFIXES = (".npz", ".joblib", ".pkl", ".png", ".gz", ".zip")


def fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    if not path.name.endswith(BINARY_SUFFIXES):
        data = data.replace(b"\r\n", b"\n")
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def shipped_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path == MANIFEST:
            continue
        if SKIP_DIRS & set(relative.parts) or path.name.endswith(SKIP_SUFFIXES) \
                or any(part.endswith(".egg-info") for part in relative.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def build() -> dict:
    return {"algorithm": "SHA-256",
            "line_endings": "text files hashed with LF",
            "excludes": "MANIFEST.json itself",
            "files": {p.relative_to(ROOT).as_posix(): fingerprint(p) for p in shipped_files()}}


def problems(prefixes: tuple = ()) -> list[str]:
    """Drift between MANIFEST.json and the tree, optionally only under `prefixes`."""
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    current = {p.relative_to(ROOT).as_posix(): p for p in shipped_files()}
    if prefixes:
        recorded = {k: v for k, v in recorded.items() if k.startswith(prefixes)}
        current = {k: v for k, v in current.items() if k.startswith(prefixes)}
    found = [f"missing: {name}" for name in sorted(set(recorded) - set(current))]
    found += [f"not in manifest: {name}" for name in sorted(set(current) - set(recorded))]
    found += [f"changed: {name}" for name in sorted(set(recorded) & set(current))
              if fingerprint(current[name])["sha256"] != recorded[name]["sha256"]]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    if parser.parse_args().check:
        found = problems()
        print("\n".join(found) or "manifest matches")
        return 1 if found else 0
    MANIFEST.write_bytes((json.dumps(build(), indent=2) + "\n").encode("utf-8"))
    print(f"wrote {MANIFEST.name}: {len(json.loads(MANIFEST.read_text(encoding='utf-8'))['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
