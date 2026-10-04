"""Validate every final dataset and resource; probe the existing Lumina engine.

Nothing here trains anything. It checks schema, vocabulary, leakage, the reply guard
(English and the new Arabic one) and how today's rules/intent router treat the new seeds.
"""
from __future__ import annotations

import collections
import json
import re
import warnings
from pathlib import Path

from . import common, final_content_c as C
from .final_common import norm_key

warnings.filterwarnings("ignore")
REQUIRED = ("id", "task", "text", "labels", "group", "lang", "source", "source_type", "license",
            "annotated_by", "clinical_validity", "verified")
NEW_INTENTS = ("GREETING", "GOODBYE", "THANKS", "ABOUT_BOT", "BOT_FEEDBACK", "DISENGAGE")


def load(folder, split):
    return [json.loads(l) for l in (folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l]


def check_datasets(root):
    from lumina.taxonomy import EMOTIONS, INTENTS
    vocab = {"emotion": set(EMOTIONS), "safety": {"NORMAL", "ELEVATED", "HIGH"},
             "intent": set(INTENTS) | set(NEW_INTENTS)}
    report = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        seen_g, seen_t, problems = {}, {}, collections.Counter()
        total = 0
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            for r in load(folder, split):
                total += 1
                for k in REQUIRED:
                    if k not in r:
                        problems[f"missing:{k}"] += 1
                if not str(r["text"]).strip():
                    problems["empty_text"] += 1
                if r["source_type"] not in common.SOURCE_TYPES + ("authored_seed",):
                    problems["bad_source_type"] += 1
                for head, allowed in vocab.items():
                    if head in r["labels"] and r["labels"][head] not in allowed:
                        problems[f"bad_label:{head}:{r['labels'][head]}"] += 1
                if seen_g.setdefault(r["group"], split) != split:
                    problems["group_leak"] += 1
                if seen_t.setdefault(norm_key(r["text"]), split) != split:
                    problems["text_leak"] += 1
            digest = common.sha256_file(folder / f"{split}.jsonl")
            if digest != manifest["files_sha256"][f"{split}.jsonl"]:
                problems["sha_mismatch"] += 1
        report[folder.name] = {"rows": total, "problems": dict(problems)}
    return report


def compile_ar():
    return {k: re.compile(v) for k, v in C.AR_PROHIBITED.items()}


def check_library(lib):
    from lumina.response import check_prohibited
    ar = compile_ar()
    fill = {"action": "Take three slow breaths.", "observation": "your sleep was shorter"}
    fill_ar = {"action": "خذ ثلاثة أنفاس بطيئة.", "observation": "نومك كان أقصر"}
    issues, combos = [], {}

    def chk(text, lang, where):
        t = text.format(**(fill if lang == "en" else fill_ar))
        if lang == "en":
            v = check_prohibited(t, ())
        else:
            v = [k for k, p in ar.items() if p.search(t)]
        if v:
            issues.append({"where": where, "lang": lang, "text": t, "violations": v})

    for lang in ("en", "ar"):
        for o in lib["openers"][lang]:
            chk(o, lang, "opener")
    for strat, langs in lib["strategies"].items():
        for lang, parts in langs.items():
            bodies = parts.get("bodies") or (parts["bodies_measured"] + parts["bodies_heard"])
            for b in bodies:
                chk(b, lang, f"{strat}.body")
                if "?" in b or "؟" in b:
                    issues.append({"where": f"{strat}.body", "lang": lang, "text": b,
                                   "violations": ["question_in_body"]})
            for q in parts["questions"]:
                chk(q, lang, f"{strat}.question")
                if q.count("?") + q.count("؟") != 1:
                    issues.append({"where": f"{strat}.question", "lang": lang, "text": q,
                                   "violations": ["not_exactly_one_question"]})
            combos[f"{strat}.{lang}"] = len(lib["openers"][lang]) * len(bodies) * len(parts["questions"])
    for intent, langs in lib["intent_replies"].items():
        for lang in ("en", "ar"):
            for t in langs[lang]:
                chk(t, lang, f"intent_reply.{intent}")
    for track, spec in lib["track_overlays"].items():
        for strat, langs in spec.items():
            if strat == "avoid":
                continue
            for lang, parts in langs.items():
                for t in parts["bodies"] + parts["questions"]:
                    chk(t, lang, f"overlay.{track}.{strat}")
    return issues, combos


def check_ar_guard():
    ar = compile_ar()
    failures = []
    for text, expected in C.AR_GUARD_TESTS:
        fired = sorted(k for k, p in ar.items() if p.search(text))
        ok = (expected in fired) if expected else not fired
        if not ok:
            failures.append({"text": text, "expected": expected, "fired": fired})
    return failures


def machine_bot_turns(raw):
    import pandas as pd
    from .build_final_real import load_dialogues, split_turns
    fl, _ = load_dialogues()
    ar = compile_ar()
    total = flagged = 0
    by_rule = collections.Counter()
    for t in fl.text:
        for bot in split_turns(t)[1::2]:
            total += 1
            hits = [k for k, p in ar.items() if p.search(bot)]
            if hits:
                flagged += 1
                by_rule.update(hits)
    return {"assistant_turns": total, "flagged_by_arabic_guard": flagged,
            "share": round(flagged / total, 4), "by_rule": dict(by_rule)}


def probe_engine(seed_path):
    from lumina.orchestrator import Lumina
    lum = Lumina.load()
    seeds = [json.loads(l) for l in seed_path.read_text(encoding="utf-8").splitlines() if l]
    order = {"NORMAL": 0, "ELEVATED": 1, "HIGH": 2, "CRISIS": 3}
    table = collections.defaultdict(lambda: collections.Counter())
    misses = []
    for s in seeds:
        try:
            r = lum.turn(text=s["text"], language=s["lang"], request_id="probe")
            got = r["safety"]["level"]
        except Exception as e:                                   # report, never hide
            got = f"ERROR:{type(e).__name__}"
        key = (s["dialect"], s["expected"])
        table[key][got] += 1
        if got != s["expected"]:
            misses.append({"text": s["text"], "dialect": s["dialect"], "expected": s["expected"],
                           "got": got})
    return {f"{d}|{e}": dict(c) for (d, e), c in sorted(table.items())}, misses


def probe_intents(root):
    from lumina.intent import classify
    rows = load(root / "intent", "test") + load(root / "intent", "val")
    rows = [r for r in rows if r["labels"]["intent"] in NEW_INTENTS]
    out = collections.defaultdict(collections.Counter)
    for r in rows:
        try:
            res = classify(r["text"])
            got = res["intent"] if isinstance(res, dict) else getattr(res, "intent", str(res))
        except Exception as e:
            got = f"ERROR:{type(e).__name__}"
        out[r["labels"]["intent"]][got] += 1
    return {k: dict(v) for k, v in out.items()}


def main(root, res, raw=None):
    report = {"datasets": check_datasets(root)}
    lib = json.loads((res / "response_library.json").read_text(encoding="utf-8"))
    issues, combos = check_library(lib)
    report["response_library"] = {"guard_issues": issues, "combinations": combos,
                                  "total_combinations": sum(combos.values())}
    report["arabic_guard_self_test_failures"] = check_ar_guard()
    report["machine_written_arabic_assistant_turns"] = machine_bot_turns(raw)
    table, misses = probe_engine(res / "safety_eval_seeds.jsonl")
    report["engine_safety_probe"] = {"by_dialect_and_expected": table, "misses": misses}
    report["engine_intent_probe_on_new_intents"] = probe_intents(root)
    (res / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    return report


if __name__ == "__main__":
    import os
    root = Path(os.environ["LUMINA_OUT"])
    rep = main(root, root.parent / "resources")
    bad = {k: v["problems"] for k, v in rep["datasets"].items() if v["problems"]}
    print("dataset problems:", bad or "none")
    print("library guard issues:", len(rep["response_library"]["guard_issues"]),
          "| combos:", rep["response_library"]["total_combinations"])
    print("AR guard self-test failures:", rep["arabic_guard_self_test_failures"])
    print("machine AR assistant turns:", rep["machine_written_arabic_assistant_turns"])
    print("safety probe:", json.dumps(rep["engine_safety_probe"]["by_dialect_and_expected"], ensure_ascii=False))
    print("misses:", len(rep["engine_safety_probe"]["misses"]))
    print("intent probe:", rep["engine_intent_probe_on_new_intents"])
