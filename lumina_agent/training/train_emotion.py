"""Train the emotion understanding head.

The emotion labels are dominated by CONTENT and NEUTRAL, so macro-F1 - not
accuracy - is the number to read in the model card. A model that answered
CONTENT every time would score about 0.35 accuracy and a macro-F1 near zero.

Where the ceiling is, and why it is not a hyperparameter problem: the English
supervision is GoEmotions mapped onto the Lumina vocabulary, and the mapping
collapses six source labels into CONTENT while splitting three adjacent ones
(annoyance / anger / disappointment) into IRRITABLE / ANGRY / FRUSTRATED. Those
three, plus the two the source manifest itself calls loose matches (CALM from
relief, MOTIVATED from excitement), are exactly the classes that score worst.
Sweeping epochs, features, learning rate, regularisation, class-weight cap,
per-class emphasis and weight averaging on validation moved macro-F1 by about
0.02 in total; the settings below are the best of that sweep. Getting materially
further means revisiting the label mapping with the raw GoEmotions files, which
are not in `data/`.

Synthetic rows supply the emotions the public sources do not cover at all
(OVERWHELMED, LONELY, LOW_ENERGY) and thin ones they barely cover (CALM,
ANXIOUS, DISTRESSED). They are train-only and weighted 0.5 by the builder, which
measurably beats both 0.25 and 1.0 on validation.
"""
import argparse

from lumina.taxonomy import EMOTIONS
from training.runner import merge_datasets, run

# The full Lumina emotion vocabulary. Every member except UNKNOWN - which exists
# only for abstention - has supervision from at least one of the three sources.
TRAINED_EMOTIONS = EMOTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--l2", type=float, default=5e-4)
    args = parser.parse_args()

    run("understanding",
        merge_datasets("emotion-en-ar",
                       ("emotion_en", "emotion_ar", "emotion_synth"),
                       ("emotion_en", "emotion_ar")),
        head_definitions={"emotion": TRAINED_EMOTIONS},
        abstain_labels={"emotion": "UNKNOWN"},
        seed=args.seed, epochs=args.epochs, max_features=args.max_features,
        target_precision=args.target_precision,
        learning_rate=args.learning_rate, l2=args.l2,
        # CONTENT outnumbers CALM by more than a hundred to one, so the default
        # cap of 4.0 left the scarce classes with almost no gradient.
        class_weight_cap=10.0,
        # Per-class emphasis was tried on SAD, FEARFUL and ANGRY and cost about
        # 0.005 macro-F1 on validation, so it is not used.
        emphasis=None,
        # Synthetic rows are weighted 0.5 by the builder; honour that.
        use_weights=True,
        notes=[
            "Emotion is not a diagnosis and is never used as one.",
            "Advisory only: lumina/decision.py does not read this head, and a test "
            "asserts the decision is identical whatever it predicts.",
            "English and Arabic public emotion sources are evaluated separately only "
            "after training; synthetic rows are train-only and weighted 0.5.",
            "The GoEmotions-to-Lumina label mapping is the binding constraint on "
            "macro-F1, not the model; see this module's docstring.",
        ])


if __name__ == "__main__":
    main()
