"""Build the emotion + communication-act dataset from DailyDialog.

DailyDialog is the only corpus in this repository that carries real, manually
annotated emotion and intention labels, so it is what the understanding models
are actually trained on. Its own train/validation/test partition is respected;
utterances that leak across that partition are removed from train so the reported
scores are not inflated by memorisation.

Both label sets are mapped into Lumina's closed vocabularies. Two of the seven
emotion mappings are lossy (disgust and surprise have no Lumina member) and are
recorded as such in the manifest rather than quietly smoothed over.
"""
from __future__ import annotations

import argparse
import collections
import json

from lumina.text import normalize, language

from .common import RAW, deduplicate, describe, sha256_file, write_dataset

SOURCE = RAW / "I17-1099.Datasets" / "EMNLP_dataset"
SPLIT_FILES = {
    "train": ("train/train/dialogues_train.txt", "train/train/dialogues_emotion_train.txt",
              "train/train/dialogues_act_train.txt"),
    "val": ("validation/validation/dialogues_validation.txt",
            "validation/validation/dialogues_emotion_validation.txt",
            "validation/validation/dialogues_act_validation.txt"),
    "test": ("test/test/dialogues_test.txt", "test/test/dialogues_emotion_test.txt",
             "test/test/dialogues_act_test.txt"),
}

# DailyDialog emotion id -> Lumina emotion.
EMOTION_MAP = {0: "NEUTRAL", 1: "ANGRY", 2: "IRRITABLE", 3: "FEARFUL",
               4: "CONTENT", 5: "SAD", 6: "CONFUSED"}
LOSSY = {"2 disgust -> IRRITABLE": "no disgust member in the Lumina vocabulary; "
                                   "nearest affective neighbour",
         "6 surprise -> CONFUSED": "no surprise member; surprise is kept as the "
                                   "disorientation it produces, not as valence"}

ACT_MAP = {1: "INFORM", 2: "QUESTION", 3: "DIRECTIVE", 4: "COMMISSIVE"}

MIN_CHARS = 3


def read_lines(path):
    return [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]


def parse_split(split):
    text_file, emotion_file, act_file = (SOURCE / p for p in SPLIT_FILES[split])
    texts = read_lines(text_file)
    emotions = read_lines(emotion_file)
    acts = read_lines(act_file)
    rows, skipped = [], collections.Counter()
    for dialogue_index, (line, emotion_line, act_line) in enumerate(zip(texts, emotions, acts)):
        utterances = [u.strip() for u in line.split("__eou__") if u.strip()]
        emotion_ids = [int(x) for x in emotion_line.split()]
        act_ids = [int(x) for x in act_line.split()]
        if not (len(utterances) == len(emotion_ids) == len(act_ids)):
            # A handful of DailyDialog lines disagree on length; an example whose
            # label may belong to a neighbouring turn is worse than no example.
            skipped["misaligned_dialogue"] += 1
            continue
        for turn, (utterance, emotion_id, act_id) in enumerate(zip(utterances, emotion_ids, act_ids)):
            if len(utterance) < MIN_CHARS:
                skipped["too_short"] += 1
                continue
            if emotion_id not in EMOTION_MAP or act_id not in ACT_MAP:
                skipped["unknown_label"] += 1
                continue
            rows.append({
                "id": f"dd_{split}_{dialogue_index:05d}_{turn:02d}",
                "task": "dialogue",
                "text": utterance,
                "labels": {"emotion": EMOTION_MAP[emotion_id], "act": ACT_MAP[act_id]},
                "group": f"dd_{split}_{dialogue_index:05d}",
                "lang": language(utterance),
                "source": "dailydialog",
                "source_type": "public_dataset",
                "license": "CC BY-NC-SA 4.0 (DailyDialog, Li et al. 2017)",
                "annotated_by": "original_dataset",
                "clinical_validity": "not_clinical",
                "verified": False,
            })
    return rows, dict(skipped)


def build(seed=42):
    splits, skipped = {}, {}
    for split in SPLIT_FILES:
        rows, split_skipped = parse_split(split)
        rows, dedup_stats = deduplicate(rows, key=lambda r: normalize(r["text"]))
        splits[split] = rows
        skipped[split] = {**split_skipped, **dedup_stats}

    # Remove from train anything that also appears in validation or test. Without
    # this the model is rewarded for memorising strings it will be scored on.
    held_out = {normalize(r["text"]) for s in ("val", "test") for r in splits[s]}
    before = len(splits["train"])
    splits["train"] = [r for r in splits["train"] if normalize(r["text"]) not in held_out]
    leaked = before - len(splits["train"])

    manifest = {
        "dataset": "dialogue",
        "purpose": "emotion and communication-act understanding",
        "built_by": "data_prep/build_dialogue.py",
        "seed": seed,
        "heads": ["emotion", "act"],
        "source": {
            "name": "DailyDialog",
            "citation": "Li et al., IJCNLP 2017 (I17-1099)",
            "local_path": str(SOURCE.relative_to(RAW.parent)),
            "license": "CC BY-NC-SA 4.0",
            "commercial_use": "restricted - non-commercial licence, see TECH_DEBT",
            "sha256": {name: sha256_file(SOURCE / SPLIT_FILES[split][i])
                       for split in SPLIT_FILES
                       for i, name in enumerate(SPLIT_FILES[split])},
        },
        "split_policy": "DailyDialog official partition; whole dialogues never "
                        "span splits; train rows duplicating any val/test "
                        "utterance are removed",
        "train_rows_removed_as_leakage": leaked,
        "label_mapping": {"emotion": EMOTION_MAP, "act": ACT_MAP},
        "lossy_mappings": LOSSY,
        "clinical_validity": "not_clinical",
        "excluded_rows": skipped,
        "distribution": describe(splits, ["emotion", "act"]),
    }
    folder, counts = write_dataset("dialogue", splits, manifest)
    return folder, counts, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    folder, counts, manifest = build(args.seed)
    print(f"wrote {folder}")
    print(json.dumps({"rows": counts,
                      "leakage_removed": manifest["train_rows_removed_as_leakage"],
                      "distribution": manifest["distribution"]}, indent=1))


if __name__ == "__main__":
    main()
