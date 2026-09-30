# Rebuild and train locally

The ZIP already contains trained artifacts. You do not need to retrain before trying the assistant.

Install developer dependencies in the VS Code terminal:

```text
python -m pip install -r requirements-dev.txt
```

Rebuild data from the included MASSIVE raw files and authored synthetic banks:

```text
python -m training.build_planning_data
```

This regenerates the JSONL partitions, Excel CSV and audit using deterministic seeds. It never uses FAQ answers as intent labels. The authoritative training inputs are JSONL files. No download is required for rebuilding because the two source languages and license are bundled.

Train both models from scratch:

```text
python -m training.train_planning_models --allow-synthetic
```

The explicit flag acknowledges unreviewed generated data. The command compares rules, logistic regression, linear SVM and Naive Bayes on validation. A candidate is used only for a language where validation supports it. It overwrites `models/intent/artifact.json` and `models/friction/artifact.json` plus their metadata. Preserve a copy first if you want to compare versions.

Test the integration and then score the frozen models:

```text
python -m pytest -q
python -m evaluation.score_planning_models
python -m evaluation.run_evaluation
python run.py interactive
```

Do not repeatedly adjust training or thresholds after looking at the same test results and then call that set an untouched test. Future substantive tuning needs a new independently labeled evaluation set.

## Add suitable data

Use planning requests with intent/friction annotations and provenance, not medical questions with answers. Missing friction is `null`, not `UNKNOWN`; UNKNOWN means the example is explicitly annotated with no recognized difficulty. Put related translations, templates, conversations and participant histories in the same family/split. Record who reviewed real labels. Synthetic examples must keep `human_reviewed: false` unless actual independent review happens.

`data/planning/seed_families.json` defines grouped synthetic train/validation/test meanings. `training_extra.json` contains training-only additions. The existing `training/train_classifiers.py` is a separate legacy entry point for fully reviewed labeled JSONL; the delivered models are built by `train_planning_models.py` above.

Model training runs on CPU and uses no API keys or pretrained language models. Results may vary slightly across library/platform versions. Runtime response scores are not calibrated probabilities.
