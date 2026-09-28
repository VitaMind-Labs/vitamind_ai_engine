# Bilingual revision validation

The included **new** model was actually trained and loaded in the agent.
It replaces the old English-only default for this revision.

- **109 tests passed**, with one Starlette/HTTPX test-client deprecation warning.
  Tests cover the original report, conversation, patient-memory, safety and API
  contracts, plus direct Arabic model input, learned vocabulary coverage, paired
  split isolation and confirmation without invented symptom evidence.
- English and Arabic CLI demonstrations completed and produced full reports.
- All 8,102 supervised records passed schema, normalized duplicate and scenario /
  translation group checks. Splits: 5,665 train / 1,219 validation / 1,218 test.
- All 17,406 source Arabic openings used for vocabulary/IDF fitting come from
  the training CSV, with exact normalized overlap against source validation/test
  removed. Numeric source labels and chatbot replies were not used for fitting.
- Word vocabulary: 40,000 terms, including 37,451 Arabic-script terms. Character
  vocabulary: 60,000 terms, including 56,722 Arabic-script terms. Both include
  Latin-script terms. Arabic inference has nonzero learned features without
  an English translation stage.
- Six authored English/Arabic attention, activation and perception examples
  reached the model and produced reports with the corresponding candidate
  patterns. The English activation example's model preference abstained at the
  existing confidence threshold; the report still used observed evidence. See
  `BILINGUAL_FLOW_CHECKS.json`. These few cases do not establish broad accuracy.

## Synthetic evaluation, not clinical performance

| Held-out partition | English accuracy | Arabic accuracy |
| --- | ---: | ---: |
| Validation | 85.55% | 84.92% |
| Test | 87.66% | 88.36% |

These figures measure the supplied synthetic routing policy with shared template
vocabulary. They are not accuracy on the user's Arabic source labels, independent
clinical validation, or proof of improvement in real conversations. The small
legacy HEALTHY category cannot establish health. The source corpus is used only
for vocabulary/IDF adaptation; it does not supply verified diagnostic targets.

Full class-level and language-specific metrics, hashes and fitting counts are in
`../models/mira_bilingual.evaluation.json`. Data checks are in
`../data/BILINGUAL_VALIDATION.json`. Exact regression output is in `TEST_RESULTS.txt`.

The original source ZIP and previous project ZIP were preserved. A copy of the
redundant original nested folder was moved outside the active project under the
task's work directory. This ZIP excludes old weights and obsolete evaluation
reports. The fresh-extraction and checksum checks are recorded in
`Mira_Bilingual_Delivery_Check.json` beside the ZIP.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/validate_bilingual_dataset.py
python run_mira.py --demo
python run_mira.py --language ar --demo
```
