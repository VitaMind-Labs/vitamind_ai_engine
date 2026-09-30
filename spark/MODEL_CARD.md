# Local planning models 2.0.0

Two supervised models were actually trained from scratch and connected to the assistant:

| Component | Selected model | Train rows | Validation rows | Deployment |
|---|---|---:|---:|---|
| Intent | Linear SVM, balanced classes | 4,227 | 834 | English and Arabic |
| Task difficulty | One-vs-rest logistic regression | 1,280 | 412 | English and Arabic |

Features are TF-IDF word unigrams/bigrams and character 3–5 grams within words. No pretrained embeddings, transformer, API call or fine-tuning is used. The models load checksummed JSON coefficients and use NumPy for inference. scikit-learn is a training/test dependency, not an inference requirement.

Logistic regression, linear SVM, Naive Bayes and the original rules were compared on validation macro F1. A single global difficulty threshold of 0.2 was selected on validation, with a precision/label-count guard. Languages need at least 20 train and validation examples and an improvement over rules; mixed-language deployment did not qualify and uses fallback rules. These engineering gates do not establish statistical or clinical reliability.

An explicit-command guard protects negated calendar writes, certain definition questions and explicit priority requests. Other supported EN/AR text uses the learned classifier. Empty vocabulary overlap falls back to rules. There is no calibrated confidence score or general guarantee of out-of-domain detection.

## Frozen held-out classifier results

| Task | Test rows | Rule macro F1 | Deployed macro F1 | Other measure |
|---|---:|---:|---:|---|
| Intent | 1079 | 0.370 | 0.627 | Accuracy 83.0% |
| Difficulty | 422 | 0.171 | 0.716 | Exact label-set accuracy 49.5% |

Intent test: 909 public examples plus 170 synthetic examples. Difficulty test: 422 synthetic examples, with no independently collected public ADHD-friction benchmark. Macro F1 averages per-class F1; it is not the percentage of conversations handled correctly. Exact label-set accuracy requires all difficulty labels to match and is substantially lower than macro F1.

Language intent macro F1: EN 0.567, AR 0.708. Difficulty macro F1: EN 0.746, AR 0.689. Per-source/per-class metrics and every held-out prediction are in `reports/planning-heldout.json`.

MASSIVE has only three mapped labels in this selection. Per-source macro F1 in the JSON report uses the union of observed true/predicted labels, so predicted out-of-scope labels also contribute. It must not be compared to a three-label-only benchmark without that qualification. No models or thresholds were changed after the final test scoring.

## Limits

- These are short-utterance classifiers, not a generative conversational model.
- Synthetic difficulty labels can reinforce the author's assumptions. No clinical efficacy, diagnostic accuracy or real-patient benefit was measured.
- Task extraction and dates still use bounded rules. Recognizing ADD_TASK does not imply correct extraction of any arbitrary sentence.
- Meaningfully different intents can share wording; class errors remain. The difficulty classifier can infer overwhelm from an ordinary Arabic agenda query. Use its output as a fallible suggestion, not a clinical fact.
- The reused journal safety model was not retrained by this update and can block benign productivity requests. It can also miss unseen risk language. See its provenance file; it is not a sole crisis detector.
- Templates, memory candidates, calendar operations and runtime safety policy remain controlled code. The assistant does not automatically contact clinicians, execute tasks, prescribe treatment or diagnose conditions.

Metadata beside each artifact records the dataset hash, validation results, exact model checksum and trained language coverage. Restart the process after retraining.
