"""Build the intent dataset from the authored seed set.

Split by *family* rather than by row: every phrasing of one underlying meaning
stays in a single split. Splitting by row here would put "I cannot focus today"
in train and "I cannot focus at all today" in test, and the resulting score would
measure string similarity, not understanding.

With a set this small some families cannot be spread across three splits without
emptying one. The manifest records exactly how many families each intent has, so
a reader can see which intents are scored on one or two held-out meanings - and
discount those numbers accordingly.
"""
from __future__ import annotations

import argparse
import collections
import json

from lumina.text import language, normalize

from .common import deduplicate, describe, stratified_group_split, write_dataset
from .intent_seed import EXCLUDED_INTENTS, EXCLUSION_REASON, SEEDS


def build(seed=42):
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

    rows, dedup_stats = deduplicate(rows, key=lambda r: normalize(r["text"]))
    assignment = stratified_group_split(rows, "intent", seed=seed)
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[assignment[row["group"]]].append(row)

    families = collections.Counter(r["group"].split(":")[0] for r in
                                   {r["group"]: r for r in rows}.values())
    manifest = {
        "dataset": "intent",
        "purpose": "Lumina intent understanding",
        "built_by": "data_prep/build_intent.py",
        "seed": seed,
        "heads": ["intent"],
        "source": {
            "name": "authored seed set",
            "path": "data_prep/intent_seed.py",
            "source_type": "authored_seed",
            "note": "No corpus in data/ carries Lumina intent labels. These "
                    "sentences were written for the purpose and are not real "
                    "patient language.",
        },
        "honest_limits": [
            "A few hundred authored sentences is a seed, not a corpus.",
            "Scores measure generalisation to unseen phrasings within an authored "
            "set - not to real patient language.",
            "Several intents have only one or two meaning-families, so their "
            "held-out score rests on one or two sentences.",
            "The model stays `candidate` until retrained on reviewed real data.",
        ],
        "excluded_intents": {"intents": list(EXCLUDED_INTENTS),
                             "reason": EXCLUSION_REASON},
        "families_per_intent": dict(families),
        "split_policy": "grouped by meaning-family, stratified by intent; no "
                        "family spans two splits",
        "deduplication": dedup_stats,
        "clinical_validity": "not_clinical",
        "distribution": describe(splits, ["intent"]),
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
                      "languages": manifest["language_distribution"]}, indent=1))


if __name__ == "__main__":
    main()
