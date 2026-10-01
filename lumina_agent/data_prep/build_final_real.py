"""Datasets derived from the real / public files (no authored sentences here).

Everything is cleaned, deduplicated and split by *group*. Synthetic material is
never mixed into these datasets, so a score measured on their val/test splits is
a score on the source data and not on sentences we wrote ourselves.
"""
from __future__ import annotations

import json
import random
import re

import numpy as np
import pandas as pd

from . import common
from .final_common import RAW_NEW, finish, norm_key, row, short_hash, split_rows

SEED = 42

# --------------------------------------------------------------------------
# GoEmotions (English emotion) -> Lumina EMOTIONS
# --------------------------------------------------------------------------
GE_MAP = {
    "neutral": "NEUTRAL",
    "joy": "CONTENT", "amusement": "CONTENT", "love": "CONTENT", "pride": "CONTENT",
    "gratitude": "CONTENT", "admiration": "CONTENT",
    "relief": "CALM", "optimism": "HOPEFUL", "excitement": "MOTIVATED",
    "confusion": "CONFUSED", "annoyance": "IRRITABLE", "anger": "ANGRY",
    "disappointment": "FRUSTRATED", "sadness": "SAD", "grief": "SAD",
    "fear": "FEARFUL", "nervousness": "ANXIOUS",
    "remorse": "DISTRESSED", "embarrassment": "DISTRESSED",
}
GE_DROPPED = ["approval", "caring", "curiosity", "desire", "disapproval", "disgust",
              "realization", "surprise"]
NEUTRAL_CAP = 8000


def goemotions_labelled():
    """One row per unique comment with a rater-agreement label, or None."""
    ge = pd.concat([pd.read_csv(RAW_NEW / f"goemotions_{i}.csv") for i in (1, 2, 3)],
                   ignore_index=True)
    classes = sorted(set(GE_MAP.values()))
    for c in classes:
        cols = [k for k, v in GE_MAP.items() if v == c]
        ge["_" + c] = ge[cols].max(axis=1)
    grouped = ge.groupby("id")
    freq = grouped[["_" + c for c in classes]].mean()
    meta = grouped[["text", "author", "link_id", "subreddit", "created_utc"]].first()
    meta["n_raters"] = grouped.size()
    meta["unclear"] = grouped["example_very_unclear"].mean()
    vals = freq.values
    order = np.sort(vals, axis=1)
    meta["top"] = order[:, -1]
    meta["second"] = order[:, -2]
    meta["label"] = [classes[i] for i in vals.argmax(axis=1)]
    keep = ((meta.n_raters >= 3) & (meta.top >= 0.5) & (meta.second < 0.5)
            & (meta.unclear < 0.5))
    stats = {"unique_comments": int(len(meta)), "kept_with_agreement": int(keep.sum())}
    return meta[keep].reset_index(), stats


def build_emotion_en():
    lab, stats = goemotions_labelled()
    rng = random.Random(SEED)
    neutral_idx = lab.index[lab.label == "NEUTRAL"].tolist()
    drop = set(rng.sample(neutral_idx, max(0, len(neutral_idx) - NEUTRAL_CAP)))
    lab = lab.drop(index=list(drop))
    rows = []
    for r in lab.itertuples():
        group = (f"ge_author:{r.author}" if r.author not in ("[deleted]", "", None)
                 else f"ge_id:{r.id}")
        rows.append(row(f"ge_{r.id}", "emotion", r.text.strip(), {"emotion": r.label},
                        group, "en", "goemotions", "public_dataset",
                        "Apache-2.0 (GoEmotions, Demszky et al. 2020) - verify before shipping",
                        "crowd_raters_majority",
                        rater_agreement=round(float(r.top), 2), n_raters=int(r.n_raters)))
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    splits = split_rows(rows, "emotion", SEED)
    manifest = {
        "purpose": "English emotion head, replacing DailyDialog for distress emotions",
        "built_by": "data_prep/build_final_real.py",
        "seed": SEED, "heads": ["emotion"],
        "source": {"name": "GoEmotions raw rater rows (goemotions_1..3.csv)",
                   "note": "Raw file has one row per rater. Aggregated per comment id; "
                           "kept when >=3 raters and exactly one Lumina class has >=50% "
                           "rater support."},
        "label_mapping": GE_MAP, "unmapped_source_labels": GE_DROPPED,
        "neutral_cap": NEUTRAL_CAP, "aggregation": stats, "deduplication": dd,
        "honest_limits": [
            "Reddit replies, median ~13 words - not first-person patient language.",
            "No data for OVERWHELMED, LONELY, LOW_ENERGY - see datasets/emotion_synth.",
            "CALM (relief) and MOTIVATED (excitement) are loose matches to the source labels.",
            "Mapping is a judgement call; review it with a clinician before relying on it."],
    }
    return finish("emotion_en", splits, manifest, ["emotion"])


# --------------------------------------------------------------------------
# Arabic tweets (emotional tone) -> Lumina EMOTIONS
# --------------------------------------------------------------------------
AR_TONE_MAP = {"none": "NEUTRAL", "anger": "ANGRY", "joy": "CONTENT", "love": "CONTENT",
               "sadness": "SAD", "fear": "FEARFUL"}
_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"@\w+")


def clean_tweet(t):
    t = _URL.sub(" ", str(t))
    t = _HANDLE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_emotion_ar():
    df = pd.read_csv(RAW_NEW / "Emotional-Tone-Dataset.csv")
    df.columns = [c.strip() for c in df.columns]
    df["LABEL"] = df["LABEL"].astype(str).str.strip()
    df = df[df.LABEL.isin(AR_TONE_MAP)]
    rows = []
    for r in df.itertuples():
        text = clean_tweet(r.TWEET)
        if len(text.split()) < 2:
            continue
        rows.append(row(f"art_{r.ID}", "emotion", text, {"emotion": AR_TONE_MAP[r.LABEL]},
                        f"art:{short_hash(norm_key(text))}", "ar", "arabic_emotional_tone_tweets",
                        "public_dataset", "unverified - confirm terms before commercial use",
                        "original_dataset",
                        label_semantics="tweet emotion class; not personal distress",
                        source_label=r.LABEL))
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    splits = split_rows(rows, "emotion", SEED)
    manifest = {
        "purpose": "Arabic coarse emotion head (Egyptian/Gulf/MSA tweets)",
        "built_by": "data_prep/build_final_real.py", "seed": SEED, "heads": ["emotion"],
        "source": {"name": "Emotional-Tone-Dataset.csv"},
        "label_mapping": AR_TONE_MAP,
        "unmapped_source_labels": ["sympathy", "surprise"], "deduplication": dd,
        "honest_limits": [
            "Tweets, often about public events - a 'sadness' row can be about football, not the writer.",
            "Mixed dialects, no dialect label, no author id (grouped by normalized text).",
            "Only NEUTRAL/ANGRY/CONTENT/SAD/FEARFUL are covered."],
    }
    return finish("emotion_ar", splits, manifest, ["emotion"])


# --------------------------------------------------------------------------
# Arabic dialect dialogues (data.tar) -> patient-side utterances only
# --------------------------------------------------------------------------
DIALECT_FIX = {"MSA": "MSA", "EGY": "EGY", "GLF": "GLF", "LEV": "LEV",
               "EG": "EGY", "EEGY": "EGY", "E-G-Y": "EGY"}
_QUOTES = '"\u201c\u201d\u00ab\u00bb '

ABOUT_BOT_AR = re.compile(
    r"(هل|ايش|إيش|شو|ايه|إيه|مين|من)\s*(أنت|انت|إنت|انتي|أنتِ)\b.{0,25}"
    r"(إنسان|انسان|بشر|روبوت|بوت|برنامج|ذكاء|حقيقي|دكتور|طبيب)"
    r"|\b(أنت|انت|إنت)\s+(إنسان|انسان|بشر|روبوت|بوت|برنامج|حقيقي)")


COND_CANON = {"ocd", "anxiety", "depression", "other", "suicide", "normal"}
# Typos found in the source column (Cyrillic/Arabic letters inside "ocd", "suicio").
COND_FIX = {"o\u0441d": "ocd", "oc\u062f": "ocd", "oocd": "ocd", "o_\u062f": "ocd",
            "o\u062f": "ocd", "suicio": "suicide"}


def canon_condition(value):
    v = str(value).strip(_QUOTES).lower()
    v = COND_FIX.get(v, v)
    return v if v in COND_CANON else None


def load_dialogues():
    fl = pd.read_csv(RAW_NEW / "flattened_data.csv")
    n0 = len(fl)
    fl["dialect"] = fl["dialect"].astype(str).str.strip(_QUOTES).str.upper()
    fl = fl[fl.dialect.isin(DIALECT_FIX)].copy()
    fl["dialect"] = fl.dialect.map(DIALECT_FIX)
    fl["chatbot"] = fl["chatbot_related_questions"].astype(str).str.strip(_QUOTES) == "True"
    fl["condition"] = fl["Label"].map(canon_condition)
    bad_label = int(fl.condition.isna().sum())
    fl = fl[fl.condition.notna()].copy()
    stats = {"rows_in_file": n0, "after_dropping_header_and_shifted_rows": len(fl) + bad_label,
             "dropped_unreadable_condition_label": bad_label, "rows_kept": len(fl)}
    return fl, stats


def split_turns(text):
    return [t.strip(_QUOTES) for t in str(text).split(" | ") if t.strip(_QUOTES)]


def build_ar_dialogue_user():
    fl, stats = load_dialogues()
    rows, about_bot = [], []
    seen_dialogue = set()
    dup_dialogues = 0
    for r in fl.itertuples():
        full_key = norm_key(r.text)
        if full_key in seen_dialogue:
            dup_dialogues += 1
            continue
        seen_dialogue.add(full_key)
        turns = split_turns(r.text)
        if len(turns) < 2:
            continue
        group = f"ard:{short_hash(norm_key(turns[0]))}"     # same opener -> same split
        dlg = short_hash(full_key, 10)
        for i in range(0, len(turns), 2):                   # even index = patient
            utt = turns[i]
            words = len(utt.split())
            if not 3 <= words <= 120:
                continue
            base = dict(dialect=r.dialect, turn_index=i // 2,
                        dialogue_condition=r.condition,
                        flag_chatbot_related_dialogue=bool(r.chatbot))
            rows.append(row(f"ard_{dlg}_{i // 2}", "ar_patient_language", utt,
                            {"dialect": r.dialect}, group, "ar", "arabic_dialect_dialogues",
                            "synthetic",
                            "undocumented - appears machine-generated; confirm terms",
                            "unknown", **base))
            if ABOUT_BOT_AR.search(utt):
                about_bot.append((utt, group, dlg, i // 2, r.dialect))
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    dd["duplicate_dialogues_skipped"] = dup_dialogues
    splits = split_rows(rows, "dialect", SEED)
    manifest = {
        "purpose": "Arabic patient-side utterances (MSA/Egyptian/Gulf/Levantine): dialect head, "
                   "Arabic corpus for self-trained embeddings, Arabic stress-test inputs",
        "built_by": "data_prep/build_final_real.py", "seed": SEED, "heads": ["dialect"],
        "source": {"name": "data.tar -> processed/flattened_data.csv",
                   "note": "Turns alternate patient|assistant. Only patient turns are kept."},
        "cleaning": stats, "deduplication": dd,
        "excluded": "Assistant turns are NOT shipped: they are machine-written, not clinician replies, "
                    "and unreviewed. A narrow Arabic guard flags a small share for diagnosis, "
                    "medication advice or episode naming (numbers in resources/validation_report.json); "
                    "the rest are simply unvetted.",
        "condition_label": "flattened_data.csv carries a text label per DIALOGUE (ocd, anxiety, "
                           "depression, other, suicide, normal); final/*.csv encodes the same six "
                           "as 0-5. Kept as dialogue_condition on every turn for traceability; "
                           "it describes the whole dialogue, not each turn, so it is not a turn label.",
        "honest_limits": [
            "Undocumented provenance; appears machine-generated, so not real patient language.",
            "Train/val/test of the original archive shared 692 identical texts; rebuilt here "
            "from the flattened file with duplicate dialogues removed and split by opening turn.",
            "Condition labels had typos (e.g. Cyrillic letters in 'ocd'); repaired or dropped.",
            "Dialect label comes from the source and was not independently checked."],
    }
    counts = finish("ar_dialogue_user", splits, manifest, ["dialect"])
    return counts, about_bot


# --------------------------------------------------------------------------
# Arabic doctor Q&A -> patient questions with topic labels (no answers)
# --------------------------------------------------------------------------
QA_FILES = ["Psychotic_Disorders.csv", "Psychological_Issues_and_Guidance.csv",
            "Personality_and_Self_Development.csv", "Neuropsychological_Conditions.csv",
            "Behavioral_Psychological_Conditions.csv"]
_SALUTE = re.compile(
    r"^(?:بسم الله الرحمن الرحيم|السلام عليكم(?: ورحمة الله(?: وبركاته)?)?|مرحبا|مرحباً|أهلا|أهلاً|"
    r"اهلا|الحمد لله|تحية طيبة|بعد التحية|دكتور|دكتورة|الدكتور|الدكتورة)[\s،,.:!؟\-]*")
_EMAIL = re.compile(r"\S+@\S+")
_PHONE = re.compile(r"\+?\d[\d\s\-]{7,}\d")


def topic_of(label):
    sub = label.split(" - ", 1)[1] if " - " in label else label
    if "ثنائي القطب" in label:
        return "BIPOLAR"
    if "الاكتئاب" in sub:
        return "DEPRESSION"
    if "القلق والتوتر" in sub:
        return "ANXIETY"
    if "المخاوف" in sub:
        return "FEARS_PHOBIA"
    if "الوساوس" in sub:
        return "OCD"
    if "الفصام" in sub or "الهلوسة" in sub or "الذهانية" in sub:
        return "PSYCHOSIS"
    if "تشتت الذهن" in sub:
        return "CONCENTRATION"
    if "ضعف الذاكرة" in sub:
        return "MEMORY"
    if "النوم" in sub:
        return "SLEEP"
    if "الحركات اللاإرادية" in sub:
        return "MOTOR_TREMOR"
    if sub.startswith("تطوير الذات"):
        return "SELF_DEVELOPMENT"
    if "الشخصية" in sub:
        return "PERSONALITY"
    if label.startswith("قضايا وإرشادات نفسية عامة"):
        return "GENERAL_GUIDANCE"
    return "OTHER"


def clean_question(text, max_words=80):
    t = _EMAIL.sub(" ", str(text))
    t = _PHONE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for _ in range(4):
        new = _SALUTE.sub("", t).strip()
        if new == t:
            break
        t = new
    return " ".join(t.split()[:max_words])


def build_ar_patient_questions():
    frames = [pd.read_csv(RAW_NEW / f)[["Question", "Hierarchical Diagnosis"]]
              for f in QA_FILES]
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    rows = []
    for r in df.itertuples(index=False):
        text = clean_question(r[0])
        if len(text.split()) < 8:
            continue
        label = str(r[1])
        topic = topic_of(label)
        rows.append(row(f"arq_{short_hash(norm_key(text))}", "ar_topic", text,
                        {"topic": topic}, f"arq:{short_hash(norm_key(text))}", "ar",
                        "arabic_doctor_qa_questions", "research_dataset",
                        "unverified - scraped consultation text; confirm terms before commercial use",
                        "forum_category",
                        label_semantics="forum category of the consultation, not a diagnosis",
                        source_category=label))
    rows, dd = common.deduplicate(rows, key=lambda x: norm_key(x["text"]))
    splits = split_rows(rows, "topic", SEED)
    manifest = {
        "purpose": "Arabic topic head over real patient questions; Arabic vocabulary source",
        "built_by": "data_prep/build_final_real.py", "seed": SEED, "heads": ["topic"],
        "source": {"name": "five Arabic doctor Q&A CSVs", "rows_in": n0},
        "cleaning": "salutations stripped, emails/phones removed, truncated to first 80 words",
        "deduplication": dd,
        "excluded": "Doctor answers are NOT shipped: 300-word clinical letters with medication "
                    "advice and religious framing; they would violate the reply guard.",
        "honest_limits": [
            "Topic = forum category, never a diagnosis. There is no ADHD category in the source.",
            "CONCENTRATION is concentration/daydreaming, not ADHD.",
            "PSYCHOSIS ~600 questions and BIPOLAR 300 - small.",
            "Questions are much longer than chat messages; truncated but still longer."],
    }
    return finish("ar_patient_questions", splits, manifest, ["topic"])


# --------------------------------------------------------------------------
# Safety addon (real part): Reddit depression/SuicideWatch + long NORMAL texts
# --------------------------------------------------------------------------
DISTRESS_EXCLUDE = re.compile(
    r"\b(?:suicid\w*|kill (?:my|him|her|your|them)self|end (?:it all|my life)|want(?:ed)? to die|"
    r"wanna die|self[- ]?harm\w*|cutting|hate myself|worthless|hopeless|depress\w*|anxiet\w*|"
    r"anxious|panic\w*|overdos\w*|abus\w*|rape\w*|murder\w*|die[sd]?|dead|death|kill\w*|"
    r"hurt myself|lonely|cry\w*)\b", re.I)


def safety_rows_real():
    rows = []
    # 1. Reddit r/depression vs r/SuicideWatch -------------------------------
    mh = pd.read_csv(RAW_NEW / "mental-health.csv", encoding_errors="replace")
    combined = pd.read_csv(RAW_NEW / "Combined_Data.csv", encoding="latin-1",
                           usecols=["statement"])
    known = {norm_key(s) for s in combined.statement.dropna().astype(str)}
    overlap = 0
    mapping = {"depression": "ELEVATED", "SuicideWatch": "HIGH"}
    len_rng = random.Random(SEED)
    for i, r in enumerate(mh.itertuples(index=False)):
        full = " ".join(str(r.text).split()[:150])
        if len(full.split()) < 5:
            continue
        key = norm_key(full)
        if key in known:
            overlap += 1
            continue
        # Cut to the same 60-130 word range as the NORMAL texts so length cannot
        # separate the classes. Costs some signal at the tail of long posts.
        text = " ".join(full.split()[:len_rng.randint(60, 130)])
        level = mapping.get(r.label)
        if level is None:
            continue
        rows.append(row(f"mh_{short_hash(key)}", "safety", text,
                        {"safety": level, "safety_signal": level},
                        f"mh:{short_hash(key)}", "en", "reddit_depression_suicidewatch",
                        "weak_supervision",
                        "unverified - Reddit-derived Kaggle dataset; confirm terms",
                        "community_of_origin",
                        label_semantics=f"derived from the subreddit '{r.label}'; "
                                        "not a clinician risk rating"))
    mh_stats = {"rows_in": int(len(mh)), "overlap_with_combined_data_dropped": overlap}

    # 2. Long NORMAL texts. Three sources, all filtered by an exclusion lexicon.
    #    They exist to break the "long text = distress" shortcut; each brings its own
    #    style confound, which the manifest states.
    lab, _ = goemotions_labelled()
    ok = lab[lab.label.isin(["NEUTRAL", "CONTENT", "HOPEFUL", "CALM"])
             & ~lab.text.str.contains(DISTRESS_EXCLUDE)]
    rng = random.Random(SEED)
    counts = {"thread": 0, "subreddit": 0, "smalltalk_window": 0}

    def _concat(comments, target):
        parts, words = [], 0
        for c in comments:
            parts.append(c.strip())
            words += len(c.split())
            if words >= target:
                break
        return " ".join(parts), words

    for link, g in ok.groupby("link_id"):            # (a) same thread
        if len(g) < 5:
            continue
        text, words = _concat(g.sort_values("created_utc").text.tolist(), rng.randint(60, 130))
        if words >= 50:
            rows.append(row(f"gt_{short_hash(link)}", "safety", text,
                            {"safety": "NORMAL", "safety_signal": "NORMAL"},
                            f"gt:{link}", "en", "goemotions_thread_concat", "weak_supervision",
                            "Apache-2.0 (GoEmotions) - verify", "exclusion_lexicon",
                            label_semantics="NORMAL = neutral/positive comments, no distress words"))
            counts["thread"] += 1
    for sub, g in ok.groupby("subreddit"):           # (b) same community, not same thread
        texts = g.text.tolist()
        rng.shuffle(texts)
        per_sub = 0
        while len(texts) >= 8 and per_sub < 25:
            take, texts = texts[:rng.randint(6, 10)], texts[10:]
            text, words = _concat(take, rng.randint(60, 130))
            if words >= 50:
                rows.append(row(f"gs_{short_hash(sub + text[:40])}", "safety", text,
                                {"safety": "NORMAL", "safety_signal": "NORMAL"},
                                f"gs:{sub}:{per_sub}", "en", "goemotions_subreddit_concat",
                                "weak_supervision", "Apache-2.0 (GoEmotions) - verify",
                                "exclusion_lexicon",
                                label_semantics="NORMAL = unrelated neutral/positive comments from one "
                                                "subreddit joined together; incoherent on purpose"))
                counts["subreddit"] += 1
                per_sub += 1
    conv = pd.read_csv(RAW_NEW / "Conversation.csv")["question"].astype(str).tolist()
    for start in range(0, len(conv) - 10, 10):       # (c) consecutive small-talk turns
        text = " ".join(t.strip() for t in conv[start:start + 10])
        if 50 <= len(text.split()) <= 150 and not DISTRESS_EXCLUDE.search(text):
            rows.append(row(f"cw_{start}", "safety", text,
                            {"safety": "NORMAL", "safety_signal": "NORMAL"},
                            f"cw:{start // 10}", "en", "conversation_csv_window",
                            "weak_supervision", "unverified - confirm terms", "exclusion_lexicon",
                            label_semantics="NORMAL = ten consecutive small-talk turns"))
            counts["smalltalk_window"] += 1
    long_rows = counts
    return rows, {"reddit": mh_stats, "long_normal_rows": long_rows}


AR_COND_TO_SAFETY = {"normal": "NORMAL", "suicide": "HIGH", "depression": "ELEVATED",
                     "anxiety": "ELEVATED", "ocd": "ELEVATED", "other": "ELEVATED"}


def safety_rows_ar_dialogues():
    """First patient turn of each dialogue, labelled from the dialogue-level condition.

    Same collapse rule as Combined_Data: a condition name is never a risk level by
    itself; only the suicide label maps to HIGH. CRISIS is never assigned here.
    """
    fl, _ = load_dialogues()
    rows, seen = [], set()
    for r in fl.itertuples():
        turns = split_turns(r.text)
        if len(turns) < 2:
            continue
        utt = turns[0]
        if not 3 <= len(utt.split()) <= 120:
            continue
        key = norm_key(utt)
        if key in seen:
            continue
        seen.add(key)
        level = AR_COND_TO_SAFETY[r.condition]
        rows.append(row(f"ards_{short_hash(key)}", "safety", utt,
                        {"safety": level, "safety_signal": level},
                        f"ard:{short_hash(key)}", "ar", "arabic_dialect_dialogues_first_turn",
                        "synthetic", "undocumented - appears machine-generated; confirm terms",
                        "dialogue_level_condition_label", dialect=r.dialect, weight=0.5,
                        label_semantics=f"derived from dialogue label '{r.condition}'; the opening "
                                        "turn may be milder than the label; not a risk rating"))
    return rows


# --------------------------------------------------------------------------
# Intent addon (real part): intent.json patterns + small talk
# --------------------------------------------------------------------------
INTENT_TAG_MAP = {
    "greeting": "GREETING", "morning": "GREETING", "afternoon": "GREETING",
    "evening": "GREETING",
    "goodbye": "GOODBYE", "night": "GOODBYE", "done": "GOODBYE",
    "thanks": "THANKS", "pandora-useful": "THANKS",
    "about": "ABOUT_BOT", "skill": "ABOUT_BOT", "creation": "ABOUT_BOT",
    "location": "ABOUT_BOT",
    "repeat": "BOT_FEEDBACK", "wrong": "BOT_FEEDBACK", "stupid": "BOT_FEEDBACK",
    "understand": "BOT_FEEDBACK", "hate-you": "BOT_FEEDBACK", "hate-me": "BOT_FEEDBACK",
    "not-talking": "DISENGAGE", "something-else": "DISENGAGE", "no-approach": "DISENGAGE",
    "sleep": "SLEEP", "stressed": "STRESS", "friends": "SOCIAL",
    "sad": "EMOTIONAL_SUPPORT", "depressed": "EMOTIONAL_SUPPORT",
    "worthless": "EMOTIONAL_SUPPORT", "anxious": "EMOTIONAL_SUPPORT",
    "scared": "EMOTIONAL_SUPPORT", "death": "EMOTIONAL_SUPPORT",
    "user-advice": "QUESTION", "mental-health-fact": "QUESTION", "learn-more": "QUESTION",
    "learn-mental-health": "QUESTION",
    "casual": "GENERAL_CONVERSATION", "ask": "GENERAL_CONVERSATION",
    "jokes": "GENERAL_CONVERSATION", "user-agree": "GENERAL_CONVERSATION",
    "neutral-response": "GENERAL_CONVERSATION",
}
INTENT_EXCLUDED_TAGS = {"suicide": "safety routing belongs to lumina/safety.py",
                        "name": "self-introduction placeholders", "help": "ambiguous",
                        "default": "single topic words", "problem": "ambiguous",
                        "no-response": "empty pattern"}
NEW_INTENTS = ["GREETING", "GOODBYE", "THANKS", "ABOUT_BOT", "BOT_FEEDBACK", "DISENGAGE"]


def intent_rows_real():
    data = json.loads((RAW_NEW / "intent.json").read_text(encoding="utf-8"))["intents"]
    rows = []
    for item in data:
        tag = item["tag"]
        label = INTENT_TAG_MAP.get(tag)
        if tag.startswith("fact-"):
            label = "QUESTION"
        if label is None:
            continue
        for k, pat in enumerate(item["patterns"]):
            pat = pat.strip()
            if len(pat.split()) < 2 and label not in ("GREETING", "GOODBYE", "THANKS"):
                continue
            if not pat:
                continue
            rows.append(row(f"ij_{tag}_{k:02d}", "intent", pat, {"intent": label},
                            f"ij:{tag if not tag.startswith('fact-') else 'facts'}", "en",
                            "intents_json_kaggle", "public_dataset",
                            "unverified - Kaggle intents file; confirm terms", "original_dataset",
                            source_tag=tag, needs_human_review=True))
    conv = pd.read_csv(RAW_NEW / "Conversation.csv")
    rng = random.Random(SEED)
    pool = []
    for i, q in enumerate(conv.question.astype(str)):
        w = len(q.split())
        if 3 <= w <= 14 and not DISTRESS_EXCLUDE.search(q):
            pool.append((i, q.strip()))
    rng.shuffle(pool)
    seen = set()
    n_small = 0
    for i, q in pool:
        key = norm_key(q)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row(f"conv_{i}", "intent", q, {"intent": "GENERAL_CONVERSATION"},
                        f"conv:{i // 25}", "en", "conversation_csv_smalltalk",
                        "public_dataset", "unverified - confirm terms", "original_dataset",
                        needs_human_review=True, weight=0.5))
        n_small += 1
        if n_small >= 200:
            break
    # Arabic "are you a robot / a real person / a doctor" questions found in patient turns.
    # (A "you keep repeating" regex was tried and dropped: it matched symptoms and
    # self-insults, not complaints about the assistant.)
    fl, _ = load_dialogues()
    ab, seen_ab = [], set()
    for r in fl.itertuples():
        for u in split_turns(r.text)[0::2]:
            if 2 <= len(u.split()) <= 14 and ABOUT_BOT_AR.search(u):
                k = norm_key(u)
                if k not in seen_ab:
                    seen_ab.add(k)
                    ab.append((u, r.dialect))
    rng.shuffle(ab)
    for j, (u, dialect) in enumerate(ab[:150]):
        core = " ".join(norm_key(u).split()[:3])
        rows.append(row(f"ardab_{short_hash(norm_key(u))}", "intent", u,
                        {"intent": "ABOUT_BOT"}, f"ardab:{short_hash(core)}", "ar",
                        "arabic_dialect_dialogues", "synthetic",
                        "undocumented - appears machine-generated; confirm terms", "regex_match",
                        dialect=dialect, needs_human_review=True))
    return rows, {"small_talk_rows": n_small, "ar_about_bot_rows": len(ab[:150]),
                  "excluded_tags": INTENT_EXCLUDED_TAGS}


def main():
    out = {}
    out["emotion_en"] = build_emotion_en()
    out["emotion_ar"] = build_emotion_ar()
    (counts, about_bot) = build_ar_dialogue_user()
    out["ar_dialogue_user"] = counts
    out["ar_patient_questions"] = build_ar_patient_questions()
    print("AR about-bot candidates:", len(about_bot))
    return out, about_bot


if __name__ == "__main__":
    main()
