# Lumina ADHD Assistant — trained local edition 1.1

A standalone English/Arabic planning assistant. It runs locally without API keys, pretrained language-model weights or fine-tuning. No VitaMind, Mira, frontend or production backend integration is included.

**This version ships with two actually trained and connected models:** a linear SVM for intent and multi-label logistic regression for task difficulties. They use word/character TF-IDF features and load local JSON artifacts at startup. Task extraction, dates, prioritization, focus suggestions and replies remain controlled code/templates; this is not an unrestricted conversational LLM.

## Run in VS Code

Extract the ZIP and open **Lumina_ADHD_Assistant_Trained** in VS Code. In Terminal → New Terminal:

```text
python -m pip install -r requirements.txt
python run.py interactive
```

For Arabic:

```text
python run.py interactive --lang AR
```

Python 3.10+ is required. Dependency installation may use the internet; inference runs offline. Try:

```text
I need to email James tomorrow.
Give me an overview of my commitments tomorrow.
```

Or:

```text
لازم أكلم البنك وأخلص التقرير وأشتري أغراض
أحتاج نقطة انطلاق واضحة.
```

`/json` shows the full response, including `analysis.intentSource` and `analysis.frictionSource`. `LOCAL_TRAINED_CLASSIFIER` means the learned model was used. A small explicit-command guard protects certain clear requests and negated writes. Unsupported mixed-language text can fall back to rules.

Clear additions in interactive mode are saved in `runtime_data/calendar.sqlite3`. This is a local calendar, not Google Calendar or Outlook. Use a separate `--patient` UUID for another local demo identity. Patient scoping is not production authentication. The terminal supplies this computer's current date/time; API callers must supply the date anchor themselves.

## What it does

- Recognizes planning requests, starting difficulty, focus requests, interruptions, completion/failure reports and rescheduling.
- Extracts supported task phrases and resolves supported relative/ISO/weekday dates. Unclear dates require clarification.
- Captures commitments phrased the way people actually phrase them: "I forgot to pay rent again", "I should call the pharmacy tomorrow", "don't let me forget to submit the form", "remind me to take my meds at 8am", and named events such as "a dentist appointment on Tuesday at 3pm". A recurring phrase is saved once and flagged `RECURRING_TASK_SAVED_ONCE`; no repeat entries are invented.
- Answers "what do I have tomorrow?" with the day's saved commitments, then one starting step.
- Attaches a deadline stated in its own clause: in "I keep putting off my taxes, they're due Friday" the Friday reaches the task. It is attached only when exactly one task could own it. With two ("call the bank, buy groceries, they're due Friday") the pronoun is not resolved by guessing — both tasks are still saved, without a due date, and the reply asks which one is due. A trailing clause never overrides a date the task already stated. The reasons are reported as `DEADLINE_CLAUSE_AMBIGUOUS_MULTIPLE_TASKS`, `DEADLINE_CLAUSE_CONFLICTS_WITH_TASK_DATE`, `DEADLINE_CLAUSE_NOT_RESOLVED` and `DEADLINE_CLAUSE_WITHOUT_TASK`; none of them stops a task being saved.
- Answers a struggle with the move that fits it. The friction classifier already names the block, so "I'm overwhelmed, I don't know where to start" is met with one item at a time, "the report feels too big" with the first slice only, "I keep putting off my taxes" with a five-minute start. Naming the struggle also names the task: "putting off my taxes", "avoiding calling the dentist" and "the report feels too big" are each saved as a commitment. A message that only points at a task ("this task", "it") is not saved; it is asked about instead, and flagged `TASK_NOT_NAMED`. These are authored templates in `lumina/adhd/executive_function/friction_support.py`, not generated text.
- Closes a task named in a completion report ("I finished the report" closes "finish the report", allowing for inflection). A completion that names nothing recognizable closes nothing.
- Ranks tasks from provided deadlines, importance, dependencies, energy and difficulty. Unknown task facts remain unknown.
- Proposes a small first step and a short focus budget, with less work when capacity is reduced.
- Uses provided recovery context and repeated structured outcomes when available.
- Runs the existing local journal safety component before coaching; it does not contact clinicians or claim a message was sent.

The intent model can still misclassify ordinary statements; an Arabic calendar query can be misread as overwhelm. Planning is no longer blocked by that, because a task named in the message is acted on whichever label the classifier produced.

The journal safety model no longer interrupts benign productivity requests. It was trained on reflective journal entries, so on planning text it fired with no interpretable cue behind it — "help me focus for the next hour" was answered with a crisis prompt. An ELEVATED level that rests on the model alone, with no lexicon cue and below the confidence bar `fuse()` already requires before trusting the model unaided, is now recorded as `UNCORROBORATED_MODEL_SIGNAL_ON_PLANNING_TEXT` and does not stop the request. Lexicon cues, CRISIS, supplied journal context and the medication boundary are unchanged. `tests/test_life_organization.py` holds both sides of this: 20 crisis phrasings in English and Arabic that must still escalate, and 10 benign planning phrasings that must not.

Cue *markers* — the lexicon's own lowest tier, covering `overwhelmed`, `deadline`, `piling up`, `exhausted`, `can't focus` — do not count as corroboration either. That vocabulary is the ordinary way an ADHD user describes a hard day, and treating it as evidence let "I'm overwhelmed, I have like 15 things to do" reach a crisis prompt. Cues that carry risk on their own (`explicit`, `urgent_wish`, `serious`, `moderate`, which includes hopelessness) are unchanged.

This remains an organizational-support prototype, not a diagnosis or emergency-response service, and the safety component is not clinically validated.

## JSON and local API

```text
python run.py analyze examples/english.json
python run.py analyze examples/arabic.json
python run.py analyze examples/trained-english.json
python run.py analyze examples/trained-arabic.json
python run.py serve
```

The API binds to `127.0.0.1:8011` with `/health`, `/ready`, `/version`, and `POST /v1/lumina/adhd/organize`. `/ready` shows the models are loaded, not that they are clinically reliable. `/version` reports both active model versions. Generated API schema documentation is at `/docs`; there is no custom UI.

JSON requests are proposals unless `calendar.commit` is true and a calendar is configured. For persisted JSON requests:

```text
python run.py analyze my-request.json --calendar runtime_data/calendar.sqlite3
```

Task operations distinguish PROPOSED from APPLIED. Reusing a committed request ID with different content is rejected. Ambiguous requests and safety handoffs do not commit task changes. Task dates are local fields, not invented UTC appointments. The service does not send emails or perform the tasks.

## Dataset and trained models

The new planning dataset has **7,844 records: 5,134 public MASSIVE records and 2,710 synthetic records**. It includes English, Arabic and a small mixed-language subset. Public data supplies calendar intent labels; synthetic examples supply the missing planning/friction meanings. All generated records are marked synthetic and not independently reviewed. The user's original English ADHD FAQ CSV remains unchanged and is not used as planning supervision.

- `data/planning/planning_dataset.csv`: readable in Excel, with UTF-8 Arabic.
- `data/planning/all.jsonl`, `train.jsonl`, `validation.jsonl`, `test.jsonl`: structured labels and provenance.
- `data/planning/seed_families.json`, `training_extra.json`: editable synthetic source meanings.
- `models/intent/` and `models/friction/`: trained JSON weights, vocabulary, checksums and metrics.
- `data/raw/massive/`: English/Arabic source data, source hashes and CC BY 4.0 license.

Read `DATASET_CARD.md`, `MODEL_CARD.md`, `TRAINING_GUIDE.md` and `THIRD_PARTY_NOTICES.md`. New wording is still challenging; more rows alone do not create general language understanding.

## Verify or retrain

```text
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m evaluation.smoke_test
python -m evaluation.run_evaluation
python -m evaluation.score_planning_models
```

The held-out classifier report is `reports/planning-heldout.json`; development integration/latency probes are `reports/evaluation.json`. Classifier scores must not be presented as end-to-end assistant accuracy. Synthetic test results are not clinical validation.

To rebuild and train from scratch using the included raw files:

```text
python -m training.build_planning_data
python -m training.train_planning_models --allow-synthetic
```

Restart the assistant after training. The explicit `--allow-synthetic` flag acknowledges that these generated labels have not been independently reviewed. Normal API requests never train a model or read the datasets.
