"""Loader for Mira_Bilingual_Dynamic_Question_Bank.json (EN + AR text included).

The bank is the pool of varied interview questions. The live planner consumes
only `free_text` entries (the chat flow is free-text); structured entries
(single_select / multi_select / slider / story_select) are ignored here.
Missing/unreadable file -> None (planner falls back to 3 hardcoded questions).
No translations are invented: both languages come straight from the file.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("vitamind.mira-ai-service.question_bank")

FREE_TEXT = "free_text"

# Leading condition -> bank modules, in preference order.
MODULES_FOR_LEADING = {
    "ADHD": ("ADHD", "UNIVERSAL", "DIFFERENTIAL"),
    "BIPOLAR_SPECTRUM": ("BIPOLAR", "UNIVERSAL", "DIFFERENTIAL"),
    "PSYCHOSIS_SPECTRUM": ("PSYCHOSIS", "UNIVERSAL", "DIFFERENTIAL"),
    None: ("UNIVERSAL", "DIFFERENTIAL"),
}


def default_bank_path() -> Path:
    override = os.environ.get("MIRA_QUESTION_BANK")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "Mira_Bilingual_Dynamic_Question_Bank.json"


@lru_cache(maxsize=1)
def load_bank(path: str | None = None) -> list[dict] | None:
    bank_path = Path(path) if path else default_bank_path()
    try:
        entries = json.loads(bank_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("question bank disabled: cannot load %s (%s)", bank_path, exc)
        return None
    free = [e for e in entries
            if e.get("input_type") == FREE_TEXT
            and (e.get("question_en") or "").strip()
            and e.get("module") not in ("SAFETY", "COMPLETION")]
    if not free:
        logger.warning("question bank disabled: no free_text entries in %s", bank_path)
        return None
    logger.info("question bank loaded: %s free_text entries from %s", len(free), bank_path)
    return free


def text_for(entry: dict, language: str | None) -> str | None:
    """Langue stricte EN/AR uniquement — pas de FR/TN, pas de traduction a la volee.
    - language ar* -> question_ar si present, sinon warning + explicit gap signal.
    - language en* -> question_en.
    """
    if language not in {"en", "ar"}:
        raise ValueError("language must be 'en' or 'ar'")
    lang = language
    if lang == "ar":
        ar_text = entry.get("question_ar", "").strip()
        if ar_text:
            return ar_text
        logger.warning("question_bank: missing question_ar for %s; no Arabic question served", entry.get("question_id"))
        return None
    # EN strict
    return entry.get("question_en", "").strip()


def describe_coverage(bank: list[dict] | None) -> dict:
    """Utilitaire pour rapport Tache 2: ou la banque couvre / manque en AR."""
    if not bank:
        return {"total_free": 0, "missing_ar": [], "by_module": {}}
    missing = [e["question_id"] for e in bank if not (e.get("question_ar") or "").strip()]
    from collections import Counter
    by_module = Counter(e.get("module") for e in bank)
    return {"total_free": len(bank), "missing_ar": missing, "by_module": dict(by_module)}
