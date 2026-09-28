# Standalone Journal Sentinel architecture

```text
English / Arabic / mixed journal text
                 |
        validate + normalize
                 |
       +---------+-----------+
       |                     |
 trained local classifier   clause-scoped safety cues
       |                     |
       +---------+-----------+
                 |
          deterministic fusion
                 |
     optional personal phrase recurrence
                 |
  support tier + fixed bilingual response + recommendations
                 |
   JSON output to a caller / metadata-only weekly summary
```

Mira is not imported, started, edited or required. This project contains no web application, frontend, API server or external model client.

## Learned model

This is a small discriminative model trained from zero using NumPy. It is not an LLM or transformer. The shared input consists of TF-IDF word unigrams, word bigrams and within-word character n-grams (lengths 3–5). Character features help with related spellings; they do not provide general language understanding.

Six logical prediction tasks share this representation:

1. Tier: `none`, `low`, `moderate`, `high`.
2. Categories: nine independent binary heads (multi-label).
3. Subject: self/other.
4. Time: current/past.
5. Negation: true/false.
6. Idiom: true/false.

Each head is a learned softmax linear classifier. Sparse stochastic gradient descent minimizes weighted cross-entropy; high-tier errors receive a 3× loss multiplier. Vocabulary and weights see training rows only. Per-head temperatures are selected using validation rows only. These temperatures are statistical adjustments on synthetic data, not clinical calibration.

The NPZ artifact uses `allow_pickle=False` and a SHA-256 integrity check. A trained model is loaded once per `JournalSentinel` instance. All text up to the 12,000-character limit is analyzed; larger inputs are rejected rather than silently truncated.

## Safety fusion

- A present, self-directed harm cue or expressed death wish receives immediate support after checking the cue's context.
- Negation, reported speech, past history and idioms are scoped to the relevant clause. Their presence elsewhere cannot cancel a separate current concern.
- Uncertainty such as “I cannot promise I won't hurt myself” is not treated as a denial.
- Concerning statements about another person can produce `moderate_flagged` with an other-person response. They are not recorded as the writer's diagnosis.
- A high model prediction with weak confidence or conflicting context becomes `moderate_flagged`, unless a current affirmative safety cue requires high support.
- Model loading or inference failures produce an explicit review-needed fallback. Current affirmative high-support cues remain high even during failure.
- A repeated moderate recommendation can become optional reflection during a three-hour cooldown; its tier and human-review recommendation stay intact. High support is never cooled down.

These finite rules have gaps. Passing supplied examples is not proof that all crises or Arabic phrasing will be understood.

The attached draft's unconditional “explicit word → high” rule was replaced with scoped checking. That avoids the draft's known false alarm for “ما أبغي أموت”. The XLM-R/ONNX plan was replaced with a from-scratch model to follow the user's no-fine-tuning requirement. Present death/absence wishes get conservative support even without evidence of intent; `high` denotes support urgency, not confirmed suicidal intent.

## Input contract

```python
agent.analyze(
    text,                         # required, non-empty string
    lang=None,                    # en or ar, or script-based selection
    entry_id=None,                # stable caller-owned ID
    now=None,                     # timezone-aware datetime; defaults to UTC now
    signature_enabled=False,
    signature_phrases=(),         # phrases defined with the user/clinician
    signature_history=(),         # saved match metadata, never typing events
    last_support_at=None,         # ISO timestamp of actually acknowledged support
    adaptive_enabled=False,
    include_audit=False,
)
```

History row format:

```json
{
  "entry_id": "journal-001",
  "timestamp": "2026-09-25T12:00:00+00:00",
  "matched_phrases": ["new projects"]
}
```

Repeated analysis of one entry counts once. The five-day window excludes future and expired records. Exact normalized phrase recurrence is available; semantic embeddings and a clinical drift score are not implemented. Co-occurrence is not a relapse diagnosis. The phrases and history are sensitive caller-managed data.

## Output contract

| Field | Meaning |
|---|---|
| `tier` | Final policy support level |
| `categories` | Text categories; can contain multiple candidates, not diagnoses |
| `subject`, `temporal`, `negated`, `is_idiom` | Context of the selected decision |
| `response` | Fixed English/Arabic template and its ID; not clinician-reviewed yet |
| `action.type` | `none`, `optional_reflection`, `support_panel`, `crisis_support` |
| `action.keep_input_editable`, `preserve_text` | Always true |
| `action.suggested_theme` | Null unless adaptive suggestions are explicitly enabled |
| `action.disclosure_required` | True whenever a theme is suggested |
| `action.motion` | Always `none` |
| `review` | Recommendation/priority; delivery is always `not_sent` |
| `signature` | Optional phrase recurrence counts |
| `model` | Artifact version, availability and nonclinical classifier score |
| `audit` | Optional cue identifiers/offsets, model context, rule context, scoped checks, reason and latency |

`model.confidence` describes the model's own tier prediction, which can differ from the final policy tier. Inspect `audit.context.tier` when debugging. No score is the probability that a person will harm themselves. Rule decisions are not assigned invented clinical probabilities.

The terminal is a development runner for this library. Contacting clinicians, choosing region-specific phone numbers, consent collection, journal storage, dashboards, appointments and UI animation belong to a future application and are not executed here.
