"""Shared helpers for turning raw corpora into governed Lumina training data.

Two rules drive everything here.

1. Provenance travels with the row (spec s27, s129). A training example that has
   lost its source, licence and clinical-validity flag cannot be audited later,
   so every builder emits those fields on every row.

2. Splits are made over *groups*, never over rows (spec s61). Reddit-style
   corpora repeat the same author many times and dialogue corpora repeat the same
   speaker across turns; splitting by row would put near-duplicates on both sides
   and make the model look far better than it is.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
DATASETS = ROOT / "datasets"

# Clinical-validity vocabulary. Nothing in this repository is "clinical".
NOT_CLINICAL = "not_clinical"

SOURCE_TYPES = ("public_dataset", "research_dataset", "human_reviewed",
                "synthetic", "weak_supervision")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_split(groups, seed=42, ratios=(0.8, 0.1, 0.1)):
    """Assign whole groups to train/val/test deterministically.

    Hashing the group id with the seed gives a stable assignment that does not
    depend on iteration order, dataset size, or a later rebuild adding rows.
    """
    train_ratio, val_ratio = ratios[0], ratios[0] + ratios[1]
    assignment = {}
    for group in groups:
        digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
        position = int.from_bytes(digest[:8], "big") / float(1 << 64)
        assignment[group] = ("train" if position < train_ratio
                             else "val" if position < val_ratio else "test")
    return assignment


def stratified_group_split(rows, head, seed=42, ratios=(0.8, 0.1, 0.1)):
    """Grouped split that also guarantees every label reaches every split.

    A purely random grouped split starves rare classes: with the safety data it
    put zero CRISIS rows in validation, which would leave the one level that
    matters most uncalibrated and unevaluated. Here groups are first bucketed by
    their most severe label, then each bucket is dealt out in hash order, so the
    rare buckets are spread across the three splits instead of landing wherever
    chance puts them.

    Groups are still never broken apart - the unit of assignment is the group.
    """
    members = collections.OrderedDict()
    for row in rows:
        members.setdefault(row["group"], []).append(row)

    # A group's stratum is the rarest label it contains, so the scarce class
    # drives placement rather than being averaged away by a common co-label.
    frequency = collections.Counter(r["labels"][head] for r in rows)
    strata = collections.defaultdict(list)
    for group, group_rows in members.items():
        stratum = min((r["labels"][head] for r in group_rows), key=lambda l: frequency[l])
        strata[stratum].append(group)

    train_ratio, val_ratio = ratios[0], ratios[0] + ratios[1]
    assignment = {}
    for stratum, groups in strata.items():
        ordered = sorted(groups, key=lambda g: hashlib.sha256(
            f"{seed}:{stratum}:{g}".encode("utf-8")).hexdigest())
        total = len(ordered)
        n_train = round(total * train_ratio)
        n_val = round(total * val_ratio)
        if total >= 3:
            # Never let rounding empty a split for a class this scarce.
            n_train = min(max(n_train, 1), total - 2)
            n_val = min(max(n_val, n_train + 1), total - 1)
        for index, group in enumerate(ordered):
            assignment[group] = ("train" if index < n_train
                                 else "val" if index < n_val else "test")
    return assignment


def deduplicate(rows, key):
    """Drop exact repeats of the same normalized text, keeping the first.

    Conflicting duplicates - the same text carrying two different labels - are
    dropped entirely rather than arbitrarily resolved: they are label noise, and
    keeping one side of the disagreement teaches the model the noise.
    """
    grouped = collections.OrderedDict()
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    kept, conflicting, duplicate = [], 0, 0
    for candidates in grouped.values():
        labels = {json.dumps(c.get("labels"), sort_keys=True) for c in candidates}
        if len(labels) > 1:
            conflicting += len(candidates)
            continue
        duplicate += len(candidates) - 1
        kept.append(candidates[0])
    return kept, {"conflicting_dropped": conflicting, "duplicates_dropped": duplicate}


def write_dataset(name, splits, manifest):
    """Write datasets/<name>/{train,val,test}.jsonl plus a manifest."""
    folder = DATASETS / name
    folder.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, rows in splits.items():
        path = folder / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
        counts[split] = len(rows)
    manifest = dict(manifest)
    manifest["rows"] = counts
    manifest["files_sha256"] = {f"{s}.jsonl": sha256_file(folder / f"{s}.jsonl")
                                for s in splits}
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return folder, counts


def read_dataset(name, split):
    path = DATASETS / name / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def label_counts(rows, head):
    return dict(collections.Counter(r["labels"][head] for r in rows))


def describe(splits, heads):
    return {split: {"rows": len(rows),
                    **{h: label_counts(rows, h) for h in heads}}
            for split, rows in splits.items()}
