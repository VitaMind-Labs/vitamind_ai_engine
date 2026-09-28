# Active bilingual model

`mira_bilingual.joblib` is the newly trained English/Arabic artifact loaded by
the agent and API. It is included and ready to run. `mira_bilingual.evaluation.json`
records training data hashes, language counts, Arabic vocabulary adaptation,
script coverage and separate validation/test results for each language.

The old English-only artifact is no longer the default and is not shipped in
this ZIP. Earlier downloads remain available as backups.

Source Arabic numeric labels were not mapped to diagnoses. Labeled training uses
synthetic English/Arabic examples, and the user's source Arabic train text is used
only for vocabulary/IDF fitting. Neither establishes clinical validity.

Run `python train_mira.py` from the project root to retrain this artifact.
Back it up first if required, then restart Mira to load the new weights.
See `../docs/TRAINING.md` and `../docs/ARABIC_DATA.md`.
