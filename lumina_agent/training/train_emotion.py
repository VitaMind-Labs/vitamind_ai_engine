"""Train the emotion + communication-act understanding heads.

Both heads share one sparse feature space and one weight matrix, which is the
practical form of the spec's shared-encoder idea: the act head's supervision
helps the emotion head without either becoming unmeasurable on its own.

The emotion labels are heavily dominated by NEUTRAL, so macro-F1 - not accuracy -
is the number to read in the model card.
"""
import argparse

from lumina.taxonomy import ACTS
from training.runner import run

# Only the emotions DailyDialog actually annotates are fitted. The rest of the
# Lumina emotion vocabulary has no supervision in this repository and would be
# dead capacity; the state engine supplies those signals instead.
TRAINED_EMOTIONS = ("NEUTRAL", "CONTENT", "SAD", "ANGRY", "IRRITABLE",
                    "FEARFUL", "CONFUSED", "UNKNOWN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--target-precision", type=float, default=0.80)
    args = parser.parse_args()

    run("understanding", "dialogue",
        head_definitions={"emotion": TRAINED_EMOTIONS, "act": ACTS},
        abstain_labels={"emotion": "UNKNOWN"},
        seed=args.seed, epochs=args.epochs, max_features=args.max_features,
        emphasis={"emotion": {"SAD": 1.5, "FEARFUL": 1.5, "ANGRY": 1.3}},
        target_precision=args.target_precision,
        use_weights=False,
        notes=[
            "Emotion is not a diagnosis and is never used as one.",
            "Only DailyDialog-annotated emotions are fitted; the remaining "
            "Lumina emotion labels have no supervision in this repository.",
            "DailyDialog is CC BY-NC-SA 4.0: non-commercial. See TECH_DEBT.",
        ])


if __name__ == "__main__":
    main()
