"""Train the intent head on the authored seed set.

Read the numbers this produces with the manifest's `honest_limits` in hand. The
training partition is under a hundred authored sentences across seventeen
classes, because a grouped split of a three-meanings-per-intent seed can only be
one meaning per partition. That is enough to prove the pipeline and to give the
orchestrator a routing signal with abstention; it is not enough to claim the
model understands patient language.

The abstention target is set high on purpose. With this little evidence the head
should decline far more often than it answers, and `UNKNOWN` routes to a generic
supportive strategy rather than a confident wrong one.
"""
import argparse

from lumina.taxonomy import INTENTS
from training.runner import run

# The intents the seed set actually covers. SAFETY and CRISIS are excluded by
# design - see data_prep/intent_seed.py - and UNKNOWN exists only for abstention.
SEEDED = tuple(i for i in INTENTS if i not in ("SAFETY", "CRISIS", "UNKNOWN"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--target-precision", type=float, default=0.85)
    args = parser.parse_args()

    run("intent", "intent",
        head_definitions={"intent": SEEDED + ("UNKNOWN",)},
        abstain_labels={"intent": "UNKNOWN"},
        seed=args.seed, epochs=args.epochs, max_features=args.max_features,
        target_precision=args.target_precision,
        use_weights=False,
        notes=[
            "Trained on an authored seed set, not on real patient language.",
            "Safety and crisis intents are deliberately not modelled here; "
            "lumina/safety.py owns that decision.",
            "Candidate only. Retrain on reviewed real data before relying on it.",
        ])


if __name__ == "__main__":
    main()
