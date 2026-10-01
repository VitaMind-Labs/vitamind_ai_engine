"""Train the intent head on the merged intent corpus.

Read the numbers this produces with the manifest's `honest_limits` in hand. The
corpus is about 1 700 authored sentences plus 700 public and synthetic rows. That
is enough to route, and not enough to claim the model understands patient
language.

What changed, and why the old numbers were what they were: the authored seed set
carried three meaning-families per intent, and the split is by family, so each
intent put one family in train, one in val and one in test. The head trained on a
single way of saying a thing and was tested on a different meaning, which scored
F1 0.00 on twelve of the seventeen topic intents. The seed set now carries
seventeen families per intent, so train keeps thirteen or fourteen.

Everything else was measured and rejected before writing more data: a converged
lbfgs solver scored below this hand-rolled SGD even with matched class weights
(the partial convergence is doing useful regularisation), averaging over five
seeds changed nothing because the fit has converged, and every wider or narrower
feature space lost to the current one - though dropping character n-grams costs
0.13 macro-F1, so they are carrying real weight. Authored meanings were the only
thing that moved the score.

Abstention still matters more than accuracy here: `UNKNOWN` routes to a generic
supportive strategy, which is a better answer than a confident wrong intent.
"""
import argparse

from lumina.taxonomy import INTENTS
from training.runner import run

# Every intent the corpus covers. SAFETY and CRISIS are excluded by design - see
# data_prep/intent_seed.py - and UNKNOWN exists only for abstention.
SEEDED = tuple(i for i in INTENTS if i not in ("SAFETY", "CRISIS", "UNKNOWN"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--max-features", type=int, default=40000)
    # Abstaining costs a generic reply; a wrong topic guess costs a reply aimed
    # at the wrong subject but still supportive, and intent never reaches the
    # safety path. So coverage is worth more here than the last points of
    # precision. On validation, 0.70 beat 0.80 on router macro-F1 (0.444 vs
    # 0.407) and cut the router's abstention rate from 40% to 30%.
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--l2", type=float, default=3e-4)
    args = parser.parse_args()

    run("intent", "intent",
        head_definitions={"intent": SEEDED + ("UNKNOWN",)},
        abstain_labels={"intent": "UNKNOWN"},
        seed=args.seed, epochs=args.epochs, max_features=args.max_features,
        target_precision=args.target_precision,
        learning_rate=args.learning_rate, l2=args.l2,
        # A thousand-odd rows do not repeat a phrasing twice, so the default
        # min-document-frequency of 2 threw away most of the vocabulary.
        min_document_frequency=1,
        # GENERAL_CONVERSATION and ABOUT_BOT carry four times the rows of a topic
        # intent. Inverse-frequency weighting alone is capped, so the topic
        # intents are lifted here as well: misrouting "I cannot sleep" to small
        # talk is the failure that reaches a patient.
        class_weight_cap=8.0,
        # Synthetic addon rows are weighted 0.5 by the builder.
        use_weights=True,
        notes=[
            "Trained on authored seeds plus public and synthetic addon rows, not "
            "on real patient language.",
            "Safety and crisis intents are deliberately not modelled here; "
            "lumina/safety.py owns that decision.",
            "Candidate only. Retrain on reviewed real data before relying on it.",
        ])


if __name__ == "__main__":
    main()
