"""Build the one intent dataset, from the authored seeds plus the addon rows.

There used to be two: `datasets/intent` (authored seeds only) and
`datasets/intent_addon` (public + synthetic rows covering the six meta intents).
Training merged them in memory, so the split each one had been given separately
was the split the model actually got, and neither manifest described the corpus
the head was fitted on. Both sources are merged here instead, before the split,
and `datasets/intent` is the whole intent corpus.

Split by *family* rather than by row: every phrasing of one underlying meaning
stays in a single split. Splitting by row here would put "I cannot focus today"
in train and "I cannot focus at all today" in test, and the resulting score would
measure string similarity, not understanding.

Each intent now carries nine or ten meaning-families (see
`intent_seed_families.py` for why that mattered), so a grouped 80/10/10 split
leaves roughly eight families in train and one in each of val and test. The
manifest records the family count per intent, so a reader can still see which
intents rest on a single held-out meaning and discount those numbers.
"""
from __future__ import annotations

import argparse
import collections
import json

from lumina.text import language, normalize

from .common import (RAW, deduplicate, describe, stratified_group_split,
                     write_dataset)
from .intent_seed import EXCLUDED_INTENTS, EXCLUSION_REASON, SEEDS

# Pooled rows of the former datasets/intent_addon: public small talk, the Kaggle
# intents patterns, and authored dialect rows for the six meta intents. Kept in
# data/ because they are a source, not a dataset.
ADDON_ROWS = RAW / "intent_addon_rows.jsonl"
ADDON_PROVENANCE = RAW / "intent_addon_rows.provenance.json"


def authored_rows():
    """One row per authored phrasing, grouped by (intent, meaning-family)."""
    rows = []
    for intent, family, english, arabic in SEEDS:
        for index, text in enumerate(list(english) + list(arabic)):
            text = text.strip()
            if not text:
                continue
            rows.append({
                "id": f"int_{intent.lower()}_{family}_{index:02d}",
                "task": "intent",
                "text": text,
                "labels": {"intent": intent},
                "group": f"{intent}:{family}",
                "lang": language(text),
                "source": "lumina_authored_seed",
                "source_type": "authored_seed",
                "license": "internal - VitaMind",
                "annotated_by": "vitamind_authors",
                "clinical_validity": "not_clinical",
                "verified": False,
                "needs_human_review": True,
                "weight": 1.0,
            })
    return rows


def addon_rows():
    """The pooled addon rows, or an empty list when the snapshot is absent."""
    if not ADDON_ROWS.exists():
        return []
    rows = []
    for line in ADDON_ROWS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Authored addon rows are templates, so they carry less weight than a
        # phrasing taken from a public corpus. The trainer reads this field.
        row["weight"] = 0.5 if row.get("source_type") == "synthetic" else 1.0
        rows.append(row)
    return rows


def build(seed=42):
    rows = authored_rows() + addon_rows()
    rows, dedup_stats = deduplicate(rows, key=lambda r: normalize(r["text"]))
    assignment = stratified_group_split(rows, "intent", seed=seed)
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[assignment[row["group"]]].append(row)

    # Count meaning-families by label, not by group prefix: the addon groups are
    # named after their source ("ij:greeting"), so a prefix count would report
    # families for "ij" and none for GREETING.
    groups_per_intent = collections.defaultdict(set)
    for row in rows:
        groups_per_intent[row["labels"]["intent"]].add(row["group"])

    addon_provenance = (json.loads(ADDON_PROVENANCE.read_text(encoding="utf-8"))
                        if ADDON_PROVENANCE.exists() else None)
    manifest = {
        "dataset": "intent",
        "purpose": "Lumina intent understanding - the whole intent corpus, one split",
        "built_by": "data_prep/build_intent.py",
        "seed": seed,
        "heads": ["intent"],
        "sources": [
            {"name": "authored seed set",
             "path": "data_prep/intent_seed.py + data_prep/intent_seed_families.py",
             "source_type": "authored_seed",
             "rows": sum(1 for r in rows if r["source_type"] == "authored_seed"),
             "note": "No corpus in data/ carries Lumina intent labels. These "
                     "sentences were written for the purpose and are not real "
                     "patient language."},
            {"name": "pooled intent addon rows",
             "path": "data/intent_addon_rows.jsonl",
             "source_type": "mixed - see per-row source_type",
             "rows": sum(1 for r in rows if r["source_type"] != "authored_seed"),
             "note": "Formerly datasets/intent_addon. Public small talk and "
                     "Kaggle intent patterns plus authored dialect rows for the "
                     "six meta intents.",
             "provenance": addon_provenance},
        ],
        "honest_limits": [
            "Under a thousand authored sentences plus seven hundred mixed rows is "
            "a seed, not a corpus.",
            "Scores measure generalisation to unseen phrasings and unseen meanings "
            "within this set - not to real patient language.",
            "The twelve Lumina topic intents are authored only; nothing in them is "
            "real patient text and none of it is clinician-labelled.",
            "Report scores per source_type: authored rows test templates, not users.",
            "Synthetic rows are down-weighted to 0.5 in training and still counted "
            "in full when scoring, so the reported numbers are not flattered by them.",
            "The model stays `candidate` until retrained on reviewed real data.",
        ],
        "excluded_intents": {"intents": list(EXCLUDED_INTENTS),
                             "reason": EXCLUSION_REASON},
        "families_per_intent": {k: len(v) for k, v in sorted(groups_per_intent.items())},
        "split_policy": "grouped by meaning-family, stratified by intent; no "
                        "family spans two splits",
        "deduplication": dedup_stats,
        "clinical_validity": "not_clinical",
        "distribution": describe(splits, ["intent"]),
        "source_type_distribution": {
            split: dict(collections.Counter(r["source_type"] for r in rows_))
            for split, rows_ in splits.items()},
        "language_distribution": {
            split: dict(collections.Counter(r["lang"] for r in rows_))
            for split, rows_ in splits.items()},
    }
    folder, counts = write_dataset("intent", splits, manifest)
    return folder, counts, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    folder, counts, manifest = build(args.seed)
    print(f"wrote {folder}")
    print(json.dumps({"rows": counts,
                      "families_per_intent": manifest["families_per_intent"],
                      "source_types": manifest["source_type_distribution"],
                      "languages": manifest["language_distribution"]}, indent=1))


if __name__ == "__main__":
    main()
