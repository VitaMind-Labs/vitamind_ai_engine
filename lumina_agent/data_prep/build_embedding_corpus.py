"""Build plain-text corpora for self-trained embeddings (PPMI-SVD / fastText-style).

Not shipped pre-built: it is ~100+ MB derived from files you already hold, and it must
be rebuilt when sources change. Output: corpus/en.txt, corpus/ar.txt, one text per
line, normalized, exact-duplicates removed, plus corpus/stats.json.

    LUMINA_RAW_NEW=<raw folder> LUMINA_OUT=<datasets folder> python -m data_prep.build_embedding_corpus

Doctor ANSWERS are included here because embeddings only learn word neighbourhoods;
nothing in this corpus is ever used to generate replies. Drop them with --no-answers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from . import common
from .final_common import RAW_NEW, norm_key
from .build_final_real import QA_FILES, clean_tweet, load_dialogues, split_turns, clean_question


def _emit(lines, path, seen):
    n = words = 0
    with open(path, "w", encoding="utf-8") as f:
        for t in lines:
            t = " ".join(str(t).split())
            if len(t.split()) < 3:
                continue
            k = norm_key(t)
            if k in seen:
                continue
            seen.add(k)
            f.write(t + "\n")
            n += 1
            words += len(t.split())
    return n, words


def main(include_answers=True):
    out = common.DATASETS.parent / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    stats = {}

    def en():
        yield from pd.read_csv(RAW_NEW / "Combined_Data.csv", encoding="latin-1",
                               usecols=["statement"]).statement.dropna()
        yield from pd.read_csv(RAW_NEW / "mental-health.csv", encoding_errors="replace").text
        ge = pd.concat([pd.read_csv(RAW_NEW / f"goemotions_{i}.csv", usecols=["id", "text"])
                        for i in (1, 2, 3)]).drop_duplicates("id")
        yield from ge.text
        yield from pd.read_csv(RAW_NEW / "Conversation.csv").question
        for item in json.loads((RAW_NEW / "intent.json").read_text(encoding="utf-8"))["intents"]:
            yield from item["patterns"]          # patterns only, never the hard-coded responses

    def ar():
        fl, _ = load_dialogues()
        for t in fl.text:
            yield from split_turns(t)[0::2]      # patient turns only
        yield from (clean_tweet(t) for t in pd.read_csv(
            RAW_NEW / "Emotional-Tone-Dataset.csv").rename(columns=str.strip).TWEET)
        for f in QA_FILES:
            df = pd.read_csv(RAW_NEW / f)
            yield from (clean_question(q, max_words=400) for q in df["Question"])
            if include_answers:
                yield from df["Answer"].astype(str)

    n, w = _emit(en(), out / "en.txt", set())
    stats["en"] = {"lines": n, "words": w}
    n, w = _emit(ar(), out / "ar.txt", set())
    stats["ar"] = {"lines": n, "words": w}
    stats["include_answers"] = include_answers
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main(include_answers="--no-answers" not in sys.argv)
