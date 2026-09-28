# Train Mira in English and Arabic

The default artifact is `models/mira_bilingual.joblib`. Both the CLI agent and
the local API load it automatically. The old English-only weights are not the
active model and are not included in this delivery.

## Normal retraining

Run these commands from the project root. Install dependencies once:

```sh
python -m pip install -r requirements.txt
python tools/validate_bilingual_dataset.py
python train_mira.py
```

The trainer uses:

- `data/mira_bilingual.jsonl` for supervised routing: native patient-style English
  and Arabic text, explicit routing labels, language tags and scenario groups.
- `data/arabic_adaptation.jsonl` for **unlabeled** training-only Arabic vocabulary
  and IDF adaptation. It does not supply ADHD/bipolar/psychosis target labels.

The output includes a trained ensemble and `models/mira_bilingual.evaluation.json`.
That JSON contains overall and per-language validation/test metrics, vocabulary
script counts, source hashes and exact supervised/unlabeled record counts.
Restart the CLI or API to reload the model; a running process retains its old copy.

## Backup and restore

Before retraining, copy the model in VS Code's Explorer or run:

```sh
python -c "from shutil import copy2; copy2('models/mira_bilingual.joblib', 'models/mira_bilingual_backup.joblib')"
```

Use a different backup name if one already exists that you need to keep.
To restore that backup, stop Mira and run:

```sh
python -c "from shutil import copy2; copy2('models/mira_bilingual_backup.joblib', 'models/mira_bilingual.joblib')"
python run_mira.py --language ar
```

To train without replacing the connected model:

```sh
python train_mira.py --output models/mira_experiment.joblib
```

This alternate filename is not automatically loaded by Mira. To omit the source
Arabic vocabulary corpus while still training on labeled English/Arabic examples:

```sh
python train_mira.py --without-arabic-corpus --output models/mira_experiment.joblib
```

## Data format

Each JSONL line is one patient-text classification example with `text`, `label`,
`language` (`en`/`ar`), `split` (`train`/`validation`/`test`), `group_id`, and
`input_type: "patient_text_bilingual_v1"`. Provenance is stored in extra fields.
Do not put assistant replies or labels inside model text. All translations,
paraphrases and related scenario variants must stay in one partition.

The six compatibility labels are ADHD, BIPOLAR, PSYCHOSIS, AMBIGUOUS, OTHER and
HEALTHY. They are routing categories. HEALTHY means sparse/nonspecific fixture
evidence here, **not proof that someone is healthy**; the agent does not treat
it as a diagnosis. This small category has very few independent held-out states.

The supplied labeled examples are synthetic and not clinician-reviewed. They
inherit an author-defined policy from `data/reviewed_projections.jsonl`; despite
that historical filename, those blueprint labels were not reviewed by a clinician.
Train/validation/test share generator assumptions and template vocabulary.
Performance on them does not establish real-world or clinical accuracy.

To regenerate the native dataset from its blueprint:

```sh
python tools/generate_bilingual_dataset.py --seed 42
```

To prepare the Arabic vocabulary corpus again from the original download:

```sh
python tools/prepare_arabic_corpus.py --zip "C:/Users/21629/Downloads/20568736.zip"
```

Change the ZIP path if it is elsewhere. Preparation overwrites the corpus and
audit, never the original ZIP. Run `python tools/validate_bilingual_dataset.py`
after changing either dataset. Only training rows fit the vectorizers or
classifiers; held-out translated groups never fit model parameters.

Training learns word and character TF-IDF features and a logistic-regression /
Complement-Naive-Bayes routing ensemble. It does not train report generation,
free-form response generation, safety rules or a general-purpose LLM. The
training code retains a legacy projection mode for old fixtures; do not use
that mode as the bilingual model's dataset.
