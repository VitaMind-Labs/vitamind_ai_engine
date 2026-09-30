# Planning dataset v1

Purpose: develop English/Arabic planning-intent and task-difficulty classifiers, not diagnose ADHD or predict clinical outcomes.

| Source | Retained records | Labels |
|---|---:|---|
| MASSIVE 1.1 selected English/Arabic utterances | 5,134 | Calendar/nonplanning intents mapped to this assistant |
| AI-authored synthetic planning examples | 2,710 | Intent and/or multi-label task difficulty |
| Total | 7,844 | Null labels mean unannotated, not negative |

| Split | Rows |
|---|---:|
| Train | 5,259 |
| Validation | 1,170 |
| Test | 1,415 |

Languages: 3,948 English, 3,803 Arabic, 93 mixed. Intent labels exist on 6,140 rows; friction labels on 2,114. Some rows support both tasks. French and Tunisian-dialect-specific datasets are not included. Arabic public data uses the ar-SA locale; generated Arabic is mainly MSA with informal forms, not validated dialect coverage.

## Sources and mapping

[MASSIVE official repository](https://github.com/alexa/massive), [dataset card](https://huggingface.co/datasets/AmazonScience/massive), [MASSIVE paper](https://arxiv.org/abs/2204.08582), [SLURP paper](https://aclanthology.org/2020.emnlp-main.588/).

The English and Arabic raw files contain 16,521 utterances each. Only selected records are used: `calendar_set` → ADD_TASK; `calendar_query` → ORGANIZE_DAY; a deterministic sample of nonplanning intents → UNKNOWN. `calendar_query` means calendar/agenda reading, not ADHD-specific prioritization. Some calendar requests omit an event or use unsupported dates; intent recognition alone must not invent those details. Original slot-annotated text is retained for provenance and future work; **this update does not train a slot extractor**.

No friction labels are inferred for public MASSIVE rows. Their `friction` value is null. The source labels were supplied by the dataset publisher; the mapping to Lumina's taxonomy has not had independent human review. Source worker identifiers are in the unchanged raw files but are not model features.

The synthetic files were authored for this task. They cover 15 intents and 13 difficulty types plus UNKNOWN. Compositions add multiple difficulties to an utterance only within its existing split. Task-object substitutions and neutral framing variants remain in the same template family. They are **not records from real patients** and are not independently reviewed. Generating more variations of one template does not add equivalent independent evidence.

## Leakage and quality controls

- Preserve MASSIVE's source train/dev/test assignment and link English/Arabic translations by source ID.
- Assign all synthetic translations and variants of a template to one split; compositions retain all component family IDs.
- Remove lower-priority entire families when exact normalized text crosses splits.
- Audit nearest-neighbor character TF-IDF similarity at 0.96; remove lower-priority families for cross-split near duplicates. Holdout rows are never moved into training.
- 122 rows were quarantined by these checks. The audit found 25 cross-split lexical near-duplicate pairs before quarantine.
- Feature vocabulary/IDF and classifier weights use train rows only; model selection, language coverage and a single global friction threshold use validation only. Test predictions are scored after model selection.

Exact text and known family overlap are absent after preparation. Nearest-neighbor lexical grouping is approximate, not proof of semantic independence. The same assistant authored the synthetic examples and the code; synthetic holdouts remain limited even with family separation. Validation was used iteratively to improve training coverage. Final test labels were not used to choose models or thresholds.

## Data format and license

JSONL records include `id`, `text`, `language`, `source`, `split`, `group_id`, `family_ids`, nullable `intent`, nullable `friction`, and label provenance. `reviewed: false` / `human_reviewed: false` truthfully record that no independent reviewer checked this adapted/generated dataset.

MASSIVE data: CC BY 4.0, with Amazon copyright and the supplied license in `data/raw/massive/LICENSE`. Original source hashes and URLs are in `SOURCE.json`. The derived selection/mapping and synthetic examples are distributed with the same CC BY 4.0 notice for this dataset bundle; underlying source attribution remains required. No endorsement by Amazon or the paper authors is implied. This data notice does not change licensing of unrelated application code or the user's older FAQ CSV.

The older `adhd_dataset.csv` is English FAQ data and is excluded from this training. Its provenance/license was not supplied, so no new rights are asserted over it.
