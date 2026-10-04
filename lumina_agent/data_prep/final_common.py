"""Helpers shared by the final-dataset builders (real + synthetic).

Rows follow the schema of datasets/*/*.jsonl already in this repo. Output goes to
datasets/ unless LUMINA_OUT is set. Raw new files are read from LUMINA_RAW_NEW
(default: data/new).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

from . import common

RAW_NEW = Path(os.environ.get("LUMINA_RAW_NEW", common.RAW / "new"))
if os.environ.get("LUMINA_OUT"):
    common.DATASETS = Path(os.environ["LUMINA_OUT"])

_AR_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_WS = re.compile(r"\s+")


def norm_key(text: str) -> str:
    """Dedup key: lowercase, strip diacritics/tatweel, unify alef/ya/ta-marbuta."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = _AR_DIAC.sub("", t)
    t = t.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))
    t = re.sub(r"[^\w\s]", " ", t)
    return _WS.sub(" ", t).strip()


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def row(id_, task, text, labels, group, lang, source, source_type, license_,
        annotated_by, **extra):
    r = {"id": id_, "task": task, "text": text, "labels": labels, "group": group,
         "lang": lang, "source": source, "source_type": source_type,
         "license": license_, "annotated_by": annotated_by,
         "clinical_validity": common.NOT_CLINICAL, "verified": False}
    r.update(extra)
    return r


def split_rows(rows, head, seed=42):
    assignment = common.stratified_group_split(rows, head, seed=seed)
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[assignment[r["group"]]].append(r)
    return out


def leakage_report(splits):
    """Groups and normalized texts must never cross splits."""
    seen_g, seen_t, problems = {}, {}, {"group": 0, "text": 0}
    for name, rows in splits.items():
        for r in rows:
            g = seen_g.setdefault(r["group"], name)
            if g != name:
                problems["group"] += 1
            t = seen_t.setdefault(norm_key(r["text"]), name)
            if t != name:
                problems["text"] += 1
    return problems


def finish(name, splits, manifest, heads):
    leak = leakage_report(splits)
    manifest = dict(manifest)
    manifest["dataset"] = name
    manifest["clinical_validity"] = common.NOT_CLINICAL
    manifest["split_policy"] = manifest.get("split_policy",
        "grouped, stratified by label; no group spans two splits")
    manifest["leakage_check"] = leak
    manifest["distribution"] = common.describe(splits, heads)
    folder, counts = common.write_dataset(name, splits, manifest)
    print(f"{name}: {counts} leakage={leak}")
    return counts, leak
