"""Train the safety-triage head.

Two deliberate asymmetries, both because under-calling risk is the error that
reaches a patient:

* CRISIS and HIGH carry an extra loss multiplier, so the optimiser pays more for
  missing them than for a false alarm.
* The abstention target is set high. When the head is not confident it returns
  UNKNOWN, and UNKNOWN is treated as ELEVATED downstream - never as NORMAL.

This head is a *signal*. The deterministic lexicon and policy rules remain the
authority on how a crisis is handled; the head can raise the fused level, never
lower it.
"""
import argparse

from lumina.taxonomy import SAFETY_LEVELS
from training.runner import run

# The head is trained on the three levels the corpus supports. CRISIS is absent
# by design (see datasets/safety/manifest.json -> why_no_learned_crisis_class):
# it is decided by the deterministic rules in lumina/safety.py, which this head
# can escalate but never overrule downwards.
#
# UNKNOWN is a prediction target for abstention only, never a training label.
TRAINING_LEVELS = ("NORMAL", "ELEVATED", "HIGH")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--target-precision", type=float, default=0.85)
    args = parser.parse_args()

    run("safety", "safety",
        head_definitions={"safety_signal": TRAINING_LEVELS + ("UNKNOWN",)},
        abstain_labels={"safety_signal": "UNKNOWN"},
        seed=args.seed, epochs=args.epochs, max_features=args.max_features,
        emphasis={"safety_signal": {"HIGH": 1.6}},
        target_precision=args.target_precision,
        notes=[
            "Signal only: deterministic rules decide crisis handling.",
            "No learned CRISIS class: the corpus holds ten distinct crisis "
            "meanings, too few to train or score one.",
            "UNKNOWN must be treated as at least ELEVATED downstream.",
            "Labels are community-derived or curated, never clinician risk ratings.",
            "Not clinically validated.",
        ])


if __name__ == "__main__":
    main()
