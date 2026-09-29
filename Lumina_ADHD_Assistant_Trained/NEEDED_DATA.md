# What data would improve the next version?

The delivered assistant now has a usable public/synthetic training dataset and trained models. The remaining gap is independent, diverse planning conversations with reliable labels—not simply more copies of the same template.

Useful additions are English/Arabic requests from different writers, annotated for intent, one or more expressed difficulties, task text spans, date expressions/reference dates, negation and ambiguity. Include ordinary requests that should not trigger safety support. Obtain separate qualified review for crisis-language tests rather than labeling clinical risk from planning intents.

For multi-turn support, include anonymized or clearly synthetic conversation state: what task was active, what changed, the user's reply, and the intended next operation. Keep related user histories and translations in one split. Mark synthetic records as synthetic and unreviewed until review actually occurs.

See `DATASET_CARD.md` for the delivered schema and limitations, and `data/evaluation/planning_annotation_template.jsonl` for editable annotation examples. These templates are not a new validation set.
