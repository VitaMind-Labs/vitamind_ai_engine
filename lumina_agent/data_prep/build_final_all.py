"""One command that builds every final dataset and resource.

    LUMINA_RAW_NEW=<folder with the raw files> python -m data_prep.build_final_all

Writes datasets/<name>/{train,val,test}.jsonl + manifest.json (or $LUMINA_OUT/...)
and resources/ next to it.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import common
from . import build_final_real as R, build_final_synth as S
from . import final_content_c as C
from .final_common import finish, norm_key, split_rows

SEED = 42

AR_GUARD_SRC = '''"""Arabic reply guard.

response.py checks rendered text against English-only regexes, so an Arabic reply
is never checked. This module adds Arabic patterns under the SAME rule names, so
every existing `prohibited` tuple (track rules included) now covers both languages.

Wire-up (once, in lumina/response.py, right after PROHIBITED_PATTERNS is defined):

    from .ar_guard import extend as _extend_arabic_guard
    _extend_arabic_guard(PROHIBITED_PATTERNS)

Patterns were written without a native clinical reviewer; review before relying on them.
"""
import re

AR_PROHIBITED = {patterns}


def extend(prohibited_patterns):
    for rule, arabic in AR_PROHIBITED.items():
        existing = prohibited_patterns.get(rule)
        if existing is None:
            prohibited_patterns[rule] = re.compile(arabic)
        else:
            prohibited_patterns[rule] = re.compile(existing.pattern + "|" + arabic,
                                                   existing.flags | re.UNICODE)
    return prohibited_patterns
'''


def build_safety_addon():
    real_rows, stats = R.safety_rows_real()
    ar_rows = R.safety_rows_ar_dialogues()
    rows = real_rows + ar_rows
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    splits = split_rows(rows, "safety", SEED)
    manifest = {
        "purpose": "Adds (1) English Reddit depression/SuicideWatch posts, (2) long NORMAL texts, and "
                   "(3) Arabic first turns of machine-generated dialogues, to datasets/safety",
        "built_by": "data_prep/build_final_all.py", "seed": SEED, "heads": ["safety"],
        "label_rule": "same collapse as Combined_Data: a condition name is never a level by itself; "
                      "suicide/SuicideWatch -> HIGH, every other condition -> ELEVATED, none -> NORMAL. "
                      "CRISIS is never assigned.",
        "composition": stats | {"arabic_first_turns": len(ar_rows)},
        "deduplication": dd,
        "honest_limits": [
            "Labels come from a subreddit or a dialogue-level tag, not from a clinician.",
            "Reddit SuicideWatch posts include venting and recovery stories; HIGH here is a language "
            "signal, never a person rating.",
            "Arabic rows are machine-generated, not patient text; report Arabic scores separately.",
            "Long NORMAL texts are concatenated neutral/positive Reddit comments, filtered by an "
            "exclusion lexicon; they fix the length confound but are not diary-style text.",
            "English Reddit text overlaps the style of Combined_Data; exact overlaps were removed."],
    }
    return finish("safety_addon", splits, manifest, ["safety"])


def main():
    out_root = common.DATASETS
    _, about_bot = R.main()
    intent_rows, intent_stats = R.intent_rows_real()
    S.build_intent_addon(intent_rows, intent_stats)
    # The addon rows are a source; this turns them plus the authored seeds into
    # the one intent dataset.
    from . import build_intent
    build_intent.build(SEED)
    build_safety_addon()
    S.build_emotion_synth()
    S.build_safety_synth()
    S.build_track_signals()
    S.build_slots()

    res = out_root.parent / "resources"
    res.mkdir(parents=True, exist_ok=True)
    (res / "response_library.json").write_text(
        json.dumps(S.response_library(), ensure_ascii=False, indent=1), encoding="utf-8")
    (res / "ar_prohibited_patterns.json").write_text(
        json.dumps({"purpose": "Arabic counterparts of PROHIBITED_PATTERNS; merge by rule name",
                    "patterns": C.AR_PROHIBITED,
                    "tests": [{"text": t, "must_fire": r} for t, r in C.AR_GUARD_TESTS]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    with open(res / "safety_eval_seeds.jsonl", "w", encoding="utf-8") as f:
        for r in S.safety_eval_rows():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # lumina/ar_guard.py is live code and already carries these patterns, so it is
    # not overwritten from here. The generated form is kept beside the other
    # resources for diffing when the pattern list changes.
    (res / "ar_guard.generated.py").write_text(AR_GUARD_SRC.format(patterns=json.dumps(
        C.AR_PROHIBITED, ensure_ascii=False, indent=4)), encoding="utf-8")
    n_rows, n_patients = S.write_trajectories(res / "trajectories")
    print(f"trajectories: {n_patients} patients, {n_rows} rows")


if __name__ == "__main__":
    main()
