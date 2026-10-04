"""Generated (synthetic) datasets and resources.

Every row is tagged source_type=synthetic, verified=false, needs_human_review=true.
These sets exist for coverage and regression. A score measured on them says how well
a model learned *our templates*, not how it will do on patients.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np

from . import common
from .final_common import finish, norm_key, row, short_hash, split_rows
from . import final_content_a as A, final_content_b as B, final_content_c as C

SEED = 42
SRC = "lumina_authored_final"
LIC = "internal - VitaMind (authored); needs clinician + native Arabic review"


def _syn(id_, task, text, labels, group, lang, dialect, source=SRC, weight=0.5, **extra):
    """An authored row, carrying the weight the trainer should give it.

    These are templates - a handful of base sentences crossed with lead-ins - so a
    row of them is not worth a row of real text. The manifests have always said
    `weight<=0.5`; the field was simply never emitted, so every loader read the
    default 1.0 and the synthetic rows counted double what was intended.
    """
    return row(id_, task, text, labels, group, lang, source, "synthetic", LIC,
               "vitamind_authors", dialect=dialect, needs_human_review=True,
               weight=weight, **extra)


def intent_synth_rows():
    rows = []
    for intent, varieties in A.INTENT_SEEDS.items():
        for key, texts in varieties.items():
            lang, dialect = A.DIALECT_LANG[key]
            for text in texts:
                k = short_hash(norm_key(text), 10)
                rows.append(_syn(f"isyn_{intent.lower()}_{k}", "intent", text,
                                 {"intent": intent}, f"isyn:{intent}:{k}", lang, dialect))
    return rows


def build_intent_addon(real_rows, real_stats):
    """Write the pooled addon rows to data/, for build_intent.py to merge.

    This used to write datasets/intent_addon, a second intent dataset with its own
    split. Training then merged the two in memory, so neither manifest described
    the corpus the head was actually fitted on. These rows are a source now:
    data_prep/build_intent.py merges them with the authored seeds and splits once.
    """
    rows = list(real_rows) + intent_synth_rows()
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    manifest = {
        "purpose": "Adds the six meta intents Lumina lacks (GREETING, GOODBYE, THANKS, ABOUT_BOT, "
                   "BOT_FEEDBACK, DISENGAGE) and more real-ish rows for existing intents",
        "built_by": "data_prep/build_final_all.py", "seed": SEED, "heads": ["intent"],
        "new_intents_needed_in_taxonomy": ["GREETING", "GOODBYE", "THANKS", "ABOUT_BOT",
                                           "BOT_FEEDBACK", "DISENGAGE"],
        "sources": {"intents_json_kaggle": "public, English, unverified licence",
                    "conversation_csv_smalltalk": "public, English, GENERAL_CONVERSATION only, capped at 200",
                    "arabic_dialect_dialogues": "machine-generated Arabic, ABOUT_BOT only",
                    "lumina_authored_final": "authored, EN + MSA + Gulf + Egyptian + Levantine + Arabizi"},
        "real_part": real_stats, "deduplication": dd,
        "excluded": {"suicide tag": "safety routing belongs to lumina/safety.py",
                     "all intents.json responses": "hard-coded hotline numbers and advice; never reuse"},
        "honest_limits": [
            "Report scores per source_type: authored rows test templates, not users.",
            "Arabic BOT_FEEDBACK exists only as authored rows; a regex over real dialogues "
            "matched symptoms ('stupid', 'recurring') and was dropped.",
            "DISENGAGE can co-occur with distress ('leave me alone'); safety runs first.",
            "Existing Lumina intents still have only authored seeds; none of this is clinician-labelled."],
        "rows": len(rows),
        "note": "Pooled rows merged by data_prep/build_intent.py into datasets/intent.",
    }
    out = common.RAW / "intent_addon_rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="\n") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    (common.RAW / "intent_addon_rows.provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"intent addon rows: {len(rows)} -> {out}")
    return len(rows), dd


def build_emotion_synth():
    rows = []
    for emo, varieties in A.EMOTION_SEEDS.items():
        for key, texts in varieties.items():
            lang, dialect = A.DIALECT_LANG[key]
            leadins = A.EN_LEADINS if lang == "en" else A.AR_LEADINS
            for base in texts:
                g = short_hash(norm_key(base), 10)
                for li in leadins:
                    if li.strip() == "today" and "today" in base:
                        continue
                    text = (li + base).strip()
                    rows.append(_syn(f"esyn_{emo.lower()}_{short_hash(norm_key(text), 10)}",
                                     "emotion", text, {"emotion": emo}, f"esyn:{emo}:{g}",
                                     lang, dialect))
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    splits = split_rows(rows, "emotion", SEED)
    manifest = {
        "purpose": "Emotions with no or thin public data: OVERWHELMED, LONELY, LOW_ENERGY, "
                   "ANXIOUS, CALM, DISTRESSED (English + MSA + Gulf)",
        "built_by": "data_prep/build_final_all.py", "seed": SEED, "heads": ["emotion"],
        "deduplication": dd,
        "use": "Add to TRAIN only, with weight<=0.5, next to emotion_en / emotion_ar. Do not report "
               "its val/test as accuracy.",
        "honest_limits": ["6-8 base sentences per emotion and language; lead-in variants are the "
                          "same idea, not new evidence.",
                          "Gulf Arabic is ~2 sentences per emotion."],
    }
    return finish("emotion_synth", splits, manifest, ["emotion"])


def build_safety_synth():
    rows = []
    for level, varieties in A.SAFETY_SEEDS.items():
        for key, texts in varieties.items():
            lang, dialect = A.DIALECT_LANG[key]
            for text in texts:
                k = short_hash(norm_key(text), 10)
                rows.append(_syn(f"ssyn_{level.lower()}_{k}", "safety", text,
                                 {"safety": level, "safety_signal": level}, f"ssyn:{k}", lang,
                                 dialect,
                                 label_semantics="authored to match the documented behaviour "
                                                 "(attribution/past-resolved -> ELEVATED, idiom -> "
                                                 "NORMAL); not a clinician risk rating"))
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    splits = split_rows(rows, "safety", SEED)
    manifest = {
        "purpose": "Short first-person distress, idioms and attributions in EN/MSA/Gulf; counters the "
                   "length confound of Combined_Data and gives the Arabic head short inputs",
        "built_by": "data_prep/build_final_all.py", "seed": SEED, "heads": ["safety"],
        "deduplication": dd,
        "excluded": "CRISIS is never a training label here; rules own it (see resources/safety_eval_seeds.jsonl).",
        "use": "Train-only supplement. Clinician must confirm every HIGH row.",
        "honest_limits": ["About 70 rows; a seed set, not a corpus.",
                          "Levels are authored judgements, unverified."],
    }
    return finish("safety_synth", splits, manifest, ["safety"])


def build_track_signals():
    rows = []
    for signal, varieties in B.SIGNALS.items():
        track = B.SIGNAL_TRACK[signal]
        for key, texts in varieties.items():
            lang, dialect = A.DIALECT_LANG[key]
            for text in texts:
                k = short_hash(norm_key(text), 10)
                rows.append(_syn(f"tsig_{signal.lower()}_{k}", "track_signal", text,
                                 {"track": track, "signal": signal}, f"tsig:{signal}:{k}", lang,
                                 dialect,
                                 label_semantics="how an experience might be worded; not evidence "
                                                 "of a condition; must never set or change a track"))
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    splits = split_rows(rows, "signal", SEED)
    manifest = {
        "purpose": "Language coverage for track-relevant signals (bipolar, ADHD, psychosis), "
                   "for the clinician-facing signal layer and regression tests",
        "built_by": "data_prep/build_final_all.py", "seed": SEED, "heads": ["track", "signal"],
        "deduplication": dd,
        "rules": ["Never use to choose or change a patient's track (spec s79, s80).",
                  "Psychosis sentences are attributed experience; replies reflect, never confirm or dispute.",
                  "Output goes to the clinician dashboard or a review flag, never to the patient as a label."],
        "honest_limits": ["~24 sentences per signal across 6 language varieties.",
                          "NONE is ordinary-day text; real confusers (baby-related sleep loss, real deadlines) are missing.",
                          "No real patient language. Expect large drops on real text."],
    }
    return finish("track_signals", splits, manifest, ["signal"])


def _render_n(rendering, template, n):
    if rendering == "digit":
        return template.format(n=n)
    if rendering == "digit_ar":
        return template.format(n=str(n).translate(B.EASTERN))
    if rendering == "word":
        return template.format(n=B.EN_NUM_WORDS[n])
    if rendering == "ar_hours":
        return template.format(h=B.AR_HOURS[n])
    if rendering == "ar_tasks":
        return template.format(h=B.AR_TASKS[n])
    raise ValueError(rendering)


def build_slots():
    rows = []
    for slot, lang, dialect, template, rendering in B.SLOT_NUMERIC:
        if slot in ("energy", "mood", "stress"):
            values = range(1, 11)
        elif slot == "sleep_hours":
            values = range(2, 10) if rendering != "digit_ar" else range(3, 11)
        elif slot == "tasks_count":
            values = range(2, 11)
        else:
            values = range(1, 9)
        if rendering == "ar_hours" or rendering == "ar_tasks":
            values = [v for v in values if v in (B.AR_HOURS if rendering == "ar_hours" else B.AR_TASKS)]
        g = f"slot:{slot}:{short_hash(template, 8)}"
        for n in values:
            text = _render_n(rendering, template, n)
            value = float(n) if slot == "sleep_hours" else int(n)
            rows.append(_syn(f"slot_{short_hash(text, 10)}", "slots", text,
                             {"primary_slot": slot, "slots": {slot: value}}, g, lang, dialect))
    for slot, value, lang, dialect, text in B.SLOT_BOOLEAN:
        rows.append(_syn(f"slot_{short_hash(text, 10)}", "slots", text,
                         {"primary_slot": slot, "slots": {slot: value}},
                         f"slot:{slot}:{short_hash(text, 8)}", lang, dialect))
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    splits = split_rows(rows, "primary_slot", SEED)
    manifest = {
        "purpose": "Text -> state slots (Lumina has no chat-to-state extraction today). Train or "
                   "unit-test a rule/tagger extractor; Western and Arabic-Indic digits, Arabic number words, "
                   "dual forms (ساعتين)",
        "built_by": "data_prep/build_final_all.py", "seed": SEED, "heads": ["primary_slot"],
        "slots": {"sleep_hours": "float hours", "energy": "0-10", "mood": "0-10", "stress": "0-10",
                  "tasks_count": "int", "alcohol_drinks": "int (English only)",
                  "meds_taken": "bool", "spending_unplanned": "bool", "social_contact": "bool"},
        "split_policy": "one template = one group; values vary inside it",
        "deduplication": dd,
        "honest_limits": ["Templated: good for a regression suite, weak as training text.",
                          "Missing: ranges ('5 to 6 hours'), fractions ('half an hour'), negations "
                          "('not 5 hours'), corrections, several slots in one message, Arabic alcohol/substance "
                          "(left out on purpose; needs a native clinical reviewer)."],
    }
    return finish("slots", splits, manifest, ["primary_slot"])


def safety_eval_rows():
    out = []
    for key, texts in A.SAFETY_EVAL_CRISIS.items():
        lang, dialect = A.DIALECT_LANG[key]
        for t in texts:
            out.append({"text": t, "expected": "CRISIS", "lang": lang, "dialect": dialect,
                        "kind": "explicit_ideation_statement"})
    for level, varieties in A.SAFETY_SEEDS.items():
        for key, texts in varieties.items():
            lang, dialect = A.DIALECT_LANG[key]
            for t in texts:
                out.append({"text": t, "expected": level, "lang": lang, "dialect": dialect,
                            "kind": "seed"})
    return out


def response_library():
    return {
        "version": "lumina-response-library-v1",
        "status": "authored; needs clinician review; Arabic needs native review",
        "composition": "reply = opener + body + question. Pick each part with an anti-repeat window "
                       "over the last 6 replies; exactly one question per reply; fill {action} and "
                       "{observation} from the decision engine; use bodies_measured only when a baseline "
                       "exists, otherwise bodies_heard.",
        "openers": C.OPENERS, "strategies": C.STRATEGIES, "intent_replies": C.INTENT_REPLIES,
        "track_overlays": C.TRACK_OVERLAYS,
        "combinations_per_strategy": "see validation_report.json",
    }


# ---------------------------------------------------------------------------
# Simulated check-in trajectories
# ---------------------------------------------------------------------------
DIMS = ("sleep_hours", "energy", "stress", "mood", "focus", "routine_stability",
        "social_connection", "task_completion")
BASE = {"sleep_hours": (7.0, 0.7), "energy": (6.0, 1.0), "stress": (4.0, 1.0), "mood": (6.0, 1.0),
        "focus": (5.8, 1.0), "routine_stability": (6.0, 1.0), "social_connection": (5.5, 1.2),
        "task_completion": (6.0, 1.0)}
ADHD_SHIFT = {"focus": -1.3, "routine_stability": -1.5, "task_completion": -1.5}
SCENARIOS = {
    "manic_prodrome": {"track": "BIPOLAR", "ramp": 5, "duration": 10,
                       "delta": {"sleep_hours": -3.0, "energy": 3.0, "mood": 1.5, "stress": 0.5,
                                 "focus": -1.0, "routine_stability": -1.5, "social_connection": 1.0,
                                 "task_completion": 0.5}},
    "depressive_slump": {"track": "BIPOLAR", "ramp": 7, "duration": 14,
                         "delta": {"sleep_hours": 2.0, "energy": -3.0, "mood": -3.0, "stress": 1.0,
                                   "focus": -2.0, "routine_stability": -1.5,
                                   "social_connection": -2.5, "task_completion": -2.5}},
    "adhd_overwhelm": {"track": "ADHD", "ramp": 4, "duration": 8,
                       "delta": {"sleep_hours": -0.8, "energy": -0.5, "stress": 2.5, "mood": -1.0,
                                 "focus": -2.5, "routine_stability": -2.0, "social_connection": -0.5,
                                 "task_completion": -3.0}},
    "psychosis_prodrome": {"track": "SCHIZOPHRENIA", "ramp": 7, "duration": 12,
                           "delta": {"sleep_hours": -2.5, "energy": -0.5, "stress": 2.5, "mood": -1.0,
                                     "focus": -2.0, "routine_stability": -2.5,
                                     "social_connection": -3.0, "task_completion": -1.5}},
}
MIX = [("stable", 40), ("false_alarm_blips", 20), ("manic_prodrome", 30), ("depressive_slump", 20),
       ("adhd_overwhelm", 25), ("psychosis_prodrome", 25)]
DAYS = 70


def simulate_trajectories(seed=SEED):
    rng = np.random.default_rng(seed)
    tracks_cycle = ["BIPOLAR", "ADHD", "SCHIZOPHRENIA"]
    rows, meta = [], []
    pid = 0
    for scenario, count in MIX:
        for k in range(count):
            pid += 1
            sc = SCENARIOS.get(scenario)
            track = sc["track"] if sc else tracks_cycle[k % 3]
            base = {d: rng.normal(*BASE[d]) for d in DIMS}
            if track == "ADHD":
                for d, s in ADHD_SHIFT.items():
                    base[d] += s
            base["sleep_hours"] = float(np.clip(base["sleep_hours"], 5.0, 8.5))
            noise_sd = {d: rng.uniform(0.5, 1.1) * (0.8 if d == "sleep_hours" else 1.0)
                        for d in DIMS}
            onset = int(rng.integers(30, 51)) if sc else None
            blips = sorted(rng.choice(np.arange(20, 65), size=2, replace=False).tolist()) \
                if scenario == "false_alarm_blips" else []
            prev = {d: 0.0 for d in DIMS}
            for day in range(DAYS):
                phase, weight = "none", 0.0
                if sc:
                    t = day - onset
                    if 0 <= t < sc["ramp"]:
                        phase, weight = "ramp", (t + 1) / sc["ramp"]
                    elif sc["ramp"] <= t < sc["ramp"] + sc["duration"]:
                        phase, weight = "episode", 1.0
                    elif sc["ramp"] + sc["duration"] <= t < sc["ramp"] + sc["duration"] + 4:
                        phase = "recovery"
                        weight = 1.0 - (t - sc["ramp"] - sc["duration"] + 1) / 4.0
                blip = day in blips
                values, all_missing = {}, rng.random() < 0.06
                for d in DIMS:
                    prev[d] = 0.3 * prev[d] + rng.normal(0, noise_sd[d])
                    v = base[d] + prev[d]
                    if sc:
                        v += weight * sc["delta"][d]
                    if blip:
                        v += {"sleep_hours": -3.0, "stress": 2.5, "energy": -1.0}.get(d, 0.0)
                    if d == "sleep_hours":
                        v = float(np.clip(v, 1.0, 14.0))
                    else:
                        v = float(np.clip(v, 0.0, 10.0))
                    if all_missing or rng.random() < 0.10:
                        values[d] = None
                    else:
                        values[d] = round(v, 1)
                rows.append({"patient_id": f"sim_{pid:03d}", "track": track, "scenario": scenario,
                             "day": day, "phase": phase, "blip": blip, **values})
            meta.append({"patient_id": f"sim_{pid:03d}", "track": track, "scenario": scenario,
                         "episode_onset_day": onset, "blip_days": blips,
                         "baseline": {d: round(base[d], 2) for d in DIMS}})
    return rows, meta


def write_trajectories(folder: Path):
    rows, meta = simulate_trajectories()
    folder.mkdir(parents=True, exist_ok=True)
    cols = ["patient_id", "track", "scenario", "day", "phase", "blip", *DIMS]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join("" if r[c] is None else str(r[c]) for c in cols))
    (folder / "trajectories.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "trajectories_meta.json").write_text(json.dumps({
        "purpose": "Simulated daily check-ins with known episode onsets, isolated bad-day blips and "
                   "missing data, to test baseline + change detection + track rules.",
        "seed": SEED, "days_per_patient": DAYS, "dimensions": list(DIMS),
        "units": "sleep_hours in hours; every other dimension 0-10 (stress: high = worse)",
        "scenarios": {k: {"track": v["track"], "ramp_days": v["ramp"], "plateau_days": v["duration"],
                          "plateau_delta": v["delta"]} for k, v in SCENARIOS.items()},
        "mix": dict(MIX),
        "phase": "none | ramp | episode | recovery (ground truth); blip=true marks an isolated bad day "
                 "that is NOT an episode",
        "missing": "~10% of values and ~6% of whole days are empty",
        "honest_limits": ["Effect sizes and ramp shapes are illustrative, not clinical estimates.",
                          "Real trajectories are messier: weekly rhythm, travel, illness, medication changes.",
                          "A detector tuned on this will look better than it is."],
        "patients": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(rows), len(meta)
