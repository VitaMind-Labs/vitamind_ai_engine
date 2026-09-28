import collections
import hashlib
import json
import random
import re
from pathlib import Path
from journal_ai.normalize import normalize
from journal_ai.schema import TRAIN_TIERS, CATEGORIES

ROOT = Path(__file__).resolve().parents[1]

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def family(row):
    # Keep self, other, past and negated variants of a meaning in one partition.
    return row["core_id"].split("_")[0]

def signature(row):
    return json.dumps({k:row[k] for k in ("tier","categories","subject","temporal","negated","is_idiom")},sort_keys=True)

def prepare(seed=42):
    source = ROOT / "data" / "source"
    rows = {name:read_jsonl(source/f"{name}.jsonl") for name in ("train","val")}
    all_rows = rows["train"] + rows["val"]
    for row in all_rows:
        if row["tier"] not in TRAIN_TIERS or any(c not in CATEGORIES for c in row["categories"]):
            raise ValueError("unknown training label")
        if row["subject"] not in ("self","other") or row["temporal"] not in ("past","current"):
            raise ValueError("unknown context label")
        if type(row["negated"]) is not bool or type(row["is_idiom"]) is not bool or not row["text"].strip():
            raise ValueError("invalid training row")
    grouped_text = collections.defaultdict(list)
    for row in all_rows:
        grouped_text[normalize(row["text"])].append(row)
    conflicts = {text for text,rs in grouped_text.items() if len({signature(r) for r in rs}) > 1 or len({family(r) for r in rs}) > 1}
    unique = [rs[0] for text,rs in sorted(grouped_text.items()) if text not in conflicts]
    strata = collections.defaultdict(list)
    for f in sorted({family(r) for r in unique}):
        strata[re.sub(r"\d+$", "", f)].append(f)
    rng = random.Random(seed)
    assignment = {}
    for families in strata.values():
        rng.shuffle(families)
        n = max(1, round(len(families)*0.15)) if len(families)>=3 else 0
        for i,f in enumerate(families):
            assignment[f] = "test" if i<n else "val" if i<2*n else "train"
    splits = {k:[r for r in unique if assignment[family(r)]==k] for k in ("train","val","test")}
    prepared = ROOT / "data" / "prepared"
    prepared.mkdir(parents=True,exist_ok=True)
    for name,rs in splits.items():
        (prepared/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rs),encoding="utf-8")
    audit = {
        "seed":seed,"source_rows":{k:len(v) for k,v in rows.items()},
        "source_sha256":{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ("train.jsonl","val.jsonl","journal-test-set.json")},
        "original_shared_core_ids":sorted({r["core_id"] for r in rows["train"]}&{r["core_id"] for r in rows["val"]}),
        "original_shared_meaning_families":sorted({family(r) for r in rows["train"]}&{family(r) for r in rows["val"]}),
        "quarantined_conflicting_texts":len(conflicts),"duplicates_removed":len(all_rows)-len(unique)-sum(len(grouped_text[t]) for t in conflicts),
        "partition_policy":"Union of supplied train/val; normalized deduplication; family-stratified meaning split. Challenge set never used to fit features, weights or temperatures.",
        "partitions":{name:{"rows":len(rs),"families":sorted({family(r) for r in rs}),"tiers":dict(collections.Counter(r["tier"] for r in rs)),"languages":dict(collections.Counter(r["lang"] for r in rs))} for name,rs in splits.items()},
        "limitations":["Synthetic single-author data.","Original val has no command-hallucination, paranoia, elevated or overload examples.","Some source high labels describe ambiguous/passive wishes; labels are preserved, not clinically endorsed.","Context variants share phrases even with family splitting; clinical validation is absent.","Uploaded 48-entry challenge set is a known regression benchmark, not a blinded clinical test."]
    }
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports"/"data-audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    (prepared/"quarantine.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for t in sorted(conflicts) for r in grouped_text[t]),encoding="utf-8")
    return splits, audit

if __name__ == "__main__":
    _, audit=prepare()
    print(json.dumps(audit,ensure_ascii=False,indent=2))
