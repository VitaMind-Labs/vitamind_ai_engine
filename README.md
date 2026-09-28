# Mira — English and Arabic assessment agent

This project now includes a **trained native English/Arabic routing model**.
Arabic patient text reaches the model in Arabic. It is no longer translated into
fixed English phrases before prediction. The model is integrated into Mira's
question selection and reports, and runs locally without an API key.

## Open and run in VS Code

Extract the ZIP and open **Mira_Clean_Project**, the folder containing
`run_mira.py`. Use Python 3.10–3.12 and open **Terminal > New Terminal**:

```sh
python -m pip install -r requirements.txt
python run_mira.py --language ar
```

For English:

```sh
python run_mira.py --language en
```

The trained weights are included. You do not need to retrain before trying it.
Startup displays the loaded model's languages: `ar, en`. If Windows recognizes
`py` instead of `python`, use `py` consistently in these commands.

Try describing an experience such as:

```text
أنا مشتت منذ الطفولة وأنسى المواعيد وأضيع أغراضي
```

| Action | English | Arabic |
| --- | --- | --- |
| Request a report | `report` / `summary` | `تقرير` / `ملخص` / `أريد تقرير` |
| Explain the question | `why` | `لماذا` |
| Skip a question | `skip` | `تخطي` |
| Explain Mira's capabilities | `what can you do` | `ماذا يمكنك أن تفعل` |
| Quit | `exit` | `خروج` |

Reports appear after at most 10 assessment answers, or immediately when requested.
Missing information is reported explicitly. Corrections update the report.
Crisis support keeps input open and does not claim that someone was contacted.

To run fictional English and Arabic demonstrations:

```sh
python run_mira.py --demo
python run_mira.py --language ar --demo
```

## What was trained

| Data | Role |
| --- | --- |
| 8,102 labeled synthetic examples: 4,050 English and 4,052 Arabic | Supervised routing training and separate held-out evaluation |
| 17,406 filtered Arabic openings from your uploaded training CSV | Vocabulary and inverse-document-frequency fitting only; source numeric labels are not used |

The supervised split has 5,665 training, 1,219 validation and 1,218 test records.
English/Arabic versions of a scenario stay in the same split. Only training
records fit model parameters. Evaluation is reported separately for both languages.

Your ZIP matches the files from the [Arabic Mental Health Dataset on Zenodo](https://zenodo.org/records/20568736).
Its published description covers anxiety, OCD, depression and suicidality, rather
than verified labels for Mira's ADHD/bipolar/psychosis targets. The files contain
numbers 0–5 without their mapping. We did not guess that mapping. Bot replies,
invalid rows and duplicate/held-out-overlapping openings are excluded from
vocabulary adaptation. See [how your Arabic data is used](docs/ARABIC_DATA.md).

The new model has learned Arabic and English text features. This is still a
**screening classifier with a guided conversation**, not a generative language
model or clinically validated diagnostic system. The supervised labels are
author-defined synthetic fixtures, not confirmed patient diagnoses. More
vocabulary does not by itself demonstrate better clinical accuracy.

## Retrain the connected model

Back up `models/mira_bilingual.joblib` if you want to retain the delivered weights.
Stop the running agent, then run:

```sh
python tools/validate_bilingual_dataset.py
python train_mira.py
python run_mira.py --language ar
```

Training automatically reads the included bilingual dataset and Arabic vocabulary
corpus, writes `models/mira_bilingual.joblib`, and creates an evaluation JSON
beside it. Restarting Mira loads that file. See [training details](docs/TRAINING.md).

## Files and development

```text
run_mira.py                 Start Mira in English or Arabic
train_mira.py               Retrain the integrated bilingual model
app.py                     Optional local HTTP API
data/mira_bilingual.jsonl   Labeled bilingual synthetic examples
data/arabic_adaptation.jsonl Filtered source Arabic openings, without labels
models/mira_bilingual.joblib Trained model used by the agent
vitamind/ml/                Training, native text inputs and model loading
vitamind/mira/              Conversation, questions and reports
vitamind/clinical/          Evidence extraction, memory and screening logic
vitamind/safety/            Independent safety-support rules
tools/                     Data preparation, generation and validation
tests/                     Active regression and bilingual integration checks
docs/                      Architecture, API, data provenance and validation
```

The English projection JSONL remains a **generator blueprint** for the new
synthetic examples; it is not the native model's training input. Historical
weights, duplicate nested project and obsolete reports are excluded from this ZIP.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

See [architecture](docs/ARCHITECTURE.md), [API usage](docs/API.md), and
[executed checks](docs/VALIDATION.md). The API is a local development service
with process-local sessions and per-session capability tokens.
