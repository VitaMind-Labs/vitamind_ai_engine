# VitaMind Journal Sentinel — AI only

A standalone English–Arabic journal analyzer, completely separate from Mira.
It includes a model **trained from scratch** on your supplied data. There are no pretrained weights, fine-tuning steps, API keys, remote inference calls, or graphical interface.

This is a working development baseline. It is not a general-purpose chatbot or a clinically validated crisis detector. Read the measured results in `MODEL_CARD.md` before interpreting its output.

## Try it in the VS Code terminal

1. Extract the ZIP and open the **VitaMind_Journal_AI** folder in VS Code.
2. Open **Terminal → New Terminal**. Use Python 3.10 or newer.
3. Run:

```text
python -m pip install -r requirements.txt
python run.py interactive
```

Write a journal entry in English or Arabic. Type `/exit` to stop.
The trained model is already included; you do not need to train before trying it.
NumPy is the only runtime dependency. Installation can require internet access; analysis and training run locally.

For full JSON output:

```text
python run.py analyze --text "I feel anxious and overwhelmed today" --lang en
python run.py analyze --file examples/journal_ar.txt --lang ar --audit
python run.py batch examples/requests.jsonl
python examples/integration.py
```

Arabic UTF-8 files are useful if your terminal keyboard or encoding is inconvenient.

## What the AI does

- Classifies journal text into `none`, `low`, `moderate`, `moderate_flagged`, or `high` support tiers.
- Combines learned text classification with English/Arabic cues, negation, past/current context, idioms and references to other people.
- Returns a fixed supportive response and structured suggestions for grounding, reflection, optional disclosed theme adaptation, or crisis support.
- Counts opted-in personal phrases over a five-day window using distinct entry IDs.
- Produces five-point weekly summaries from analysis metadata, retaining earlier high classifications after edits.

The AI **returns recommendations**. A future application would implement panels, themes, storage or clinician workflows. `delivery_status` is always `not_sent`; no one is contacted. It never diagnoses bipolar disorder, ADHD or psychosis, selects treatment, or blocks writing.

## Use it from Python

```python
from journal_ai import JournalSentinel

agent = JournalSentinel()  # Load the trained weights once.
result = agent.analyze(
    "I feel anxious and overwhelmed today",
    lang="en",
    entry_id="journal-001",
    include_audit=True,
)
print(result["tier"])
print(result["response"]["text"])
print(result["review"])
```

Use `lang="ar"` for Arabic response templates. Mixed-language input is accepted. If `lang` is omitted, script proportions choose the response language; callers can override it.
Use this example from the project directory, or install in editable mode with `python -m pip install -e .`.

## Train your model again, from scratch

```text
python -m training.train
python -m training.evaluate --strict
python -m unittest discover -s tests -v
```

Training starts with zero weights each time. It does not fine-tune any existing model.
The command regenerates meaning-separated data partitions, learns vocabulary and TF-IDF statistics from the training partition, trains the prediction tasks, and writes:

```text
journal_ai/models/journal-linear/config.json
journal_ai/models/journal-linear/weights.npz
```

Restart your Python process after retraining to load the new weights. Keep both model files together. Do not edit the NPZ file by hand.
For reproducible runs, the default seed is 42. Exact numeric reproducibility can vary across NumPy/platform versions.

## Files

| Folder/file | Purpose |
|---|---|
| `journal_ai/` | Normalization, trained inference, rules, fusion, signatures, responses and reports |
| `journal_ai/models/journal-linear/` | Actual trained model and feature configuration |
| `training/` | Data preparation, from-scratch training and evaluation |
| `data/source/` | Your three uploaded datasets, copied unchanged |
| `data/prepared/` | Deduplicated, meaning-separated training/validation/test partitions |
| `reports/` | Data audit and measured model/pipeline performance |
| `tests/` | Safety, integration, data separation and reporting regressions |
| `examples/` | English/Arabic inputs and integration example |
| `ARCHITECTURE.md` | How the AI works and its input/output contract |
| `MODEL_CARD.md` | Training details, results and limitations |

`moderate_flagged` is a policy outcome, not an extra training label. The four supervised tier labels remain as supplied.

## Weekly summary

Call `journal_ai.reporting.weekly_report(analyses, lang="en")`, or put a JSON array of analysis results in a UTF-8 file and run:

```text
python run.py report saved-analyses.json --lang en
```

The report has five fixed, data-derived bullets, numeric counts and text markers. It does not need raw journal text or a language-generation model. Give each journal a stable `entry_id`; repeated analyses of the same entry are deduplicated. The caller owns persistence and access control.

## Important integration behavior

Journal text is never executed or used as instructions. The library performs no networking and saves no entries automatically. Audit output is opt-in and omits the raw text, but still contains sensitive analysis metadata; keep it on the server when integrating.

The library is synchronous and stateless. A future application should debounce analysis around 2.5 seconds and run it on save, maintain the last acknowledged support-panel time, and supply saved signature history. Input remains editable at every tier. High support is never suppressed by the three-hour moderate-panel cooldown. The model's absence produces a visible review-needed fallback rather than a silent `none` result.

Personal signatures and adaptive theme suggestions default to off. The caller must manage consent and clinician-defined phrases. Suggested colors are design choices, not treatments. Template text and safety rules require bilingual clinical review before real patient use.
