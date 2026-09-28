"""Build the safety-triage dataset.

Safety is the one head where being wrong is expensive, so the sources are kept
separable and honestly labelled rather than blended into a single confident pile.

Three sources, three very different evidentiary weights:

* VitaMind_Journal_AI curated tiers - human-authored, bilingual EN/AR, and the
  only source containing genuine CRISIS-level language. Highest weight.
* "Combined Data" statements - labelled by the community a post came from, not by
  a clinician. That makes it *weak supervision*: it says "this text reads like
  distress", never "this person is at risk". Largest by volume.
* Therapy adverse-outcome justifications - first-person statements attached to a
  recorded adverse event.

Critically, the condition names in the second source (Depression, Bipolar, ...)
are NOT carried into the safety labels. They are collapsed into ELEVATED, because
a diagnosis is not a risk level and this model must never learn to treat one as
the other. The condition names are used separately in build_condition.py, where
they are clearly marked non-diagnostic.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys

from lumina.text import normalize, language

from .common import (RAW, ROOT, deduplicate, describe, sha256_file,
                     stratified_group_split, write_dataset)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

COMBINED = RAW / "Combined Data.csv" / "Combined Data.csv"
ADVERSE = RAW / "adverse_outcomes.csv"
JOURNAL = ROOT / "VitaMind_Journal_AI" / "data" / "prepared"

# Community label -> safety level. Everything that is a *condition* name becomes
# ELEVATED; none of them imply a level of their own.
COMBINED_MAP = {"Normal": "NORMAL", "Anxiety": "ELEVATED", "Stress": "ELEVATED",
                "Depression": "ELEVATED", "Bipolar": "ELEVATED",
                "Personality disorder": "ELEVATED", "Suicidal": "HIGH"}

# Journal tier -> safety level. "high" is the curated crisis tier.
JOURNAL_MAP = {"none": "NORMAL", "low": "ELEVATED", "moderate": "ELEVATED",
               "moderate_flagged": "HIGH", "high": "CRISIS"}

SELF_HARM_EVENTS = {"death_by_suicide", "suicide_attempt",
                    "non_suicidal_self_injury", "intensification_suicidal_ideation"}

MIN_CHARS = 8
MAX_CHARS = 4000
FILLER = "no adverse event occurred."

# Every row carries two labels.
#
# `safety`        - the 4-level ground truth, kept for auditing the fusion layer.
# `safety_signal` - the 3 levels the learned head is actually trained on.
#
# They differ because the corpus cannot support a learned CRISIS class: all
# CRISIS rows come from the curated journal set, which contains only ten distinct
# *meanings* of crisis language (the rest are surface variants of those ten). A
# head trained on that would be scored against one or two unseen sentences, which
# is a number with no predictive value. So CRISIS detection stays where the spec
# puts it anyway - in the deterministic rules (lumina/safety.py) - and the learned
# head contributes a calibrated distress signal that can raise a level, never
# lower one.
SIGNAL_COLLAPSE = {"NORMAL": "NORMAL", "ELEVATED": "ELEVATED",
                   "HIGH": "HIGH", "CRISIS": "HIGH"}


def load_combined():
    rows, skipped = [], collections.Counter()
    with open(COMBINED, encoding="utf-8") as handle:
        for index, record in enumerate(csv.DictReader(handle)):
            text = (record.get("statement") or "").strip()
            status = (record.get("status") or "").strip()
            if not text or len(text) < MIN_CHARS:
                skipped["empty_or_too_short"] += 1
                continue
            if len(text) > MAX_CHARS:
                skipped["too_long"] += 1
                continue
            if status not in COMBINED_MAP:
                skipped["unmapped_status"] += 1
                continue
            rows.append({
                "id": f"cd_{index:06d}",
                "task": "safety",
                "text": text,
                "labels": {"safety": COMBINED_MAP[status]},
                "group": "cd:" + normalize(text)[:120],
                "lang": language(text),
                "source": "combined_mental_health_statements",
                "source_type": "weak_supervision",
                "license": "see datasets/safety/manifest.json - provenance unverified",
                "annotated_by": "community_of_origin",
                "label_semantics": "derived from the community/self-report label "
                                   f"{status!r}; not a clinician risk rating",
                "clinical_validity": "not_clinical",
                "verified": False,
                "weight": 1.0,
            })
    return rows, dict(skipped)


def load_journal():
    rows, skipped = [], collections.Counter()
    for split in ("train", "val", "test"):
        path = JOURNAL / f"{split}.jsonl"
        if not path.exists():
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            tier = record.get("tier")
            text = (record.get("text") or "").strip()
            if tier not in JOURNAL_MAP or len(text) < MIN_CHARS:
                skipped["unmapped_or_short"] += 1
                continue
            rows.append({
                "id": f"jn_{split}_{index:05d}",
                "task": "safety",
                "text": text,
                "labels": {"safety": JOURNAL_MAP[tier]},
                # Keep every variant of one meaning (self/other, past, negated)
                # inside a single split, exactly as the journal trainer does.
                "group": "jn:" + str(record.get("core_id", "")).split("_")[0],
                "lang": record.get("lang") or language(text),
                "source": "vitamind_journal_ai_curated",
                "source_type": "human_reviewed",
                "license": "internal - VitaMind",
                "annotated_by": "vitamind_authors",
                "label_semantics": f"journal tier {tier!r} mapped to a safety level",
                "clinical_validity": "not_clinical",
                "verified": True,
                # Curated bilingual crisis language is the scarcest and most
                # trustworthy evidence available; it is oversampled in training.
                "weight": 3.0,
            })
    return rows, dict(skipped)


def load_adverse():
    rows, skipped = [], collections.Counter()
    with open(ADVERSE, encoding="utf-8") as handle:
        for index, record in enumerate(csv.DictReader(handle)):
            text = (record.get("internal_justification") or "").strip()
            if record.get("occurred") != "True":
                skipped["no_event"] += 1
                continue
            if not text or text.lower() == FILLER or len(text) < MIN_CHARS:
                skipped["filler_or_short"] += 1
                continue
            event = record.get("event_type", "")
            rows.append({
                "id": f"ao_{index:05d}",
                "task": "safety",
                "text": text,
                "labels": {"safety": "HIGH" if event in SELF_HARM_EVENTS else "ELEVATED"},
                "group": "ao:" + str(record.get("pairing_id", index)),
                "lang": language(text),
                "source": "therapy_adverse_outcomes",
                "source_type": "research_dataset",
                "license": "see datasets/safety/manifest.json - provenance unverified",
                "annotated_by": "original_dataset",
                "label_semantics": f"adverse event {event!r} recorded as occurred",
                "clinical_validity": "not_clinical",
                "verified": False,
                "weight": 1.5,
            })
    return rows, dict(skipped)


def build(seed=42):
    combined, combined_skipped = load_combined()
    journal, journal_skipped = load_journal()
    adverse, adverse_skipped = load_adverse()
    everything = combined + journal + adverse

    everything, dedup_stats = deduplicate(everything, key=lambda r: normalize(r["text"]))
    for row in everything:
        row["labels"]["safety_signal"] = SIGNAL_COLLAPSE[row["labels"]["safety"]]

    # Stratify on the full 4-level truth so the scarce CRISIS meanings are still
    # spread across splits and remain auditable, even though the learned head
    # never sees CRISIS as its own class.
    assignment = stratified_group_split(everything, "safety", seed=seed)
    splits = {"train": [], "val": [], "test": []}
    for row in everything:
        splits[assignment[row["group"]]].append(row)

    by_source = collections.Counter(r["source"] for r in everything)
    manifest = {
        "dataset": "safety",
        "purpose": "safety triage signal (NORMAL / ELEVATED / HIGH / CRISIS)",
        "built_by": "data_prep/build_safety.py",
        "seed": seed,
        "heads": ["safety_signal"],
        "audit_label": "safety",
        "not_a_diagnosis": "Condition names in the source labels are collapsed to "
                           "ELEVATED. This model rates language risk, never a person.",
        "model_is_a_signal_not_a_gate": "Deterministic lexicon rules remain the "
                                        "authority on crisis handling; this head "
                                        "raises, never lowers, the fused level.",
        "why_no_learned_crisis_class": "All CRISIS rows come from the curated "
                                       "journal set, which holds only ten distinct "
                                       "crisis meanings; a held-out split contains "
                                       "one or two. The learned head is therefore "
                                       "trained on safety_signal (3 levels) and "
                                       "CRISIS is decided by deterministic rules.",
        "sources": {
            "combined_mental_health_statements": {
                "path": str(COMBINED.relative_to(RAW.parent)),
                "sha256": sha256_file(COMBINED),
                "source_type": "weak_supervision",
                "label_mapping": COMBINED_MAP,
                "caveat": "labels reflect the community a statement came from",
                "excluded": combined_skipped,
            },
            "vitamind_journal_ai_curated": {
                "path": str(JOURNAL.relative_to(RAW.parent)),
                "source_type": "human_reviewed",
                "label_mapping": JOURNAL_MAP,
                "note": "only source of CRISIS examples and of Arabic coverage",
                "excluded": journal_skipped,
            },
            "therapy_adverse_outcomes": {
                "path": str(ADVERSE.relative_to(RAW.parent)),
                "sha256": sha256_file(ADVERSE),
                "source_type": "research_dataset",
                "self_harm_events": sorted(SELF_HARM_EVENTS),
                "excluded": adverse_skipped,
            },
        },
        "rows_by_source": dict(by_source),
        "split_policy": "grouped split stratified by rarest safety level: "
                        "journal meaning-families, adverse-outcome pairings and "
                        "normalized statement text are each kept whole within "
                        "one split, and every level reaches every split",
        "deduplication": dedup_stats,
        "clinical_validity": "not_clinical",
        "requires_human_review": True,
        "distribution": describe(splits, ["safety", "safety_signal"]),
        "crisis_meaning_groups": {
            split: len({r["group"] for r in rows if r["labels"]["safety"] == "CRISIS"})
            for split, rows in splits.items()},
        "language_distribution": {
            split: dict(collections.Counter(r["lang"] for r in rows))
            for split, rows in splits.items()},
    }
    folder, counts = write_dataset("safety", splits, manifest)
    return folder, counts, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    folder, counts, manifest = build(args.seed)
    print(f"wrote {folder}")
    print(json.dumps({"rows": counts, "by_source": manifest["rows_by_source"],
                      "distribution": manifest["distribution"],
                      "languages": manifest["language_distribution"]}, indent=1))


if __name__ == "__main__":
    main()
