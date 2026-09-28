# Integrated bilingual architecture

```text
English or Arabic patient message
                 |
        Independent safety guard
                 |
  Evidence extraction + question-context answers
                 |
    Present / absent / unknown / conflicting memory
                 |
  Native patient text + native-language present-evidence anchors
                 |
    Shared trained English/Arabic routing ensemble
                 |
  Confirmation/follow-up question OR full screening report
```

`MiraAgent.respond` calls `IntegratedModel.assess(state, text=...)`.
For Arabic sessions, Arabic text and Arabic evidence anchors reach the learned
vectorizers directly. There is no intermediate translation to English. Short
yes/no replies use the currently asked question to update evidence; native
evidence anchors carry that meaning to the classifier.

The model reads text even when the finite phrase extractor finds no known
symptom. A sufficiently strong model preference can prioritize a confirmation
question. It **never inserts that predicted symptom into patient memory**.
Report observations and candidate patterns still require reported/confirmed
evidence; unknown answers remain unknown and denials are not overwritten.

The word and character TF-IDF vocabularies contain Arabic and Latin features.
Two logistic-regression models and one Complement-Naive-Bayes model vote on
routing. The six categories are not confirmed diagnoses or calibrated disease
probabilities. Thresholds remain engineering defaults, not clinical calibration.

| Component | File |
| --- | --- |
| Native-language model input | `vitamind/ml/bilingual_text.py` |
| Training and language-specific evaluation | `vitamind/ml/train.py` |
| Default trained model loading | `vitamind/ml/classifier.py` |
| Model-to-agent integration | `vitamind/ml/integrated.py` |
| Questions and conversation | `vitamind/mira/questions.py`, `agent.py`, `interview.py` |
| Readable and structured reports | `vitamind/mira/reporting.py` |
| Evidence memory and extraction | `vitamind/clinical/` |
| Independent safety-support rules | `vitamind/safety/detector.py` |

Reports retain the earlier completion fix: at most ten assessment answers or
an explicit request produces a complete readable report, with gaps identified.
Later corrections refresh it. Greetings and capability questions do not consume
assessment answers. Urgent support keeps typing available and does not send
notifications or force routine report completion.

This is a guided assessment conversation, not a generative language model.
Feature extraction, speaker/negation handling, templates and the safety guard
are still bounded rules. The new classifier does not establish clinical
reliability, robust general conversation, native-speaker validation or a medical
diagnosis. Notification delivery, treatment decisions, persistent patient
storage and production identity remain outside this local prototype.
