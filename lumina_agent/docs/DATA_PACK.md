# Lumina data: layout and provenance

Originally a separate `lumina_final_data/` drop. It has been merged into the
project; that folder is gone and this file is the record of what came from it.

## 1. Where everything lives

| Folder | Holds | Rule |
|---|---|---|
| `data/` | raw and pooled **sources** | Never read at training time. Inputs to `data_prep/`. |
| `datasets/` | the **training datasets**, one folder per dataset | `train/val/test.jsonl` + `manifest.json`. The only thing trainers read. |
| `data_prep/` | every builder, raw → dataset | Rebuild: `LUMINA_RAW_NEW=data/new python -m data_prep.build_final_all` |
| `resources/` | fixtures and reference data | Not training data: reply library, Arabic prohibited patterns, CRISIS eval seeds, simulated trajectories, validation report. |
| `artifacts/models/` | trained weights + model cards | Plus `artifacts/registry.json`. |

One dataset per task. There is no second copy of anything: `datasets/intent_addon`
was folded into `datasets/intent` (its rows are now the source
`data/intent_addon_rows.jsonl`), and the duplicated `lumina_final_data/datasets`
tree was byte-identical to `datasets/` and was deleted.

`data/counsel_chat_qa.jsonl` (3 511 Context/Response counselling pairs) arrived
misnamed as `intent.json` inside the addon dataset folder. It is **not** intent
data and nothing reads it — the `intent.json` the builders want is the Kaggle
intents file, expected in `data/new/`. Kept as a source: real first-person text,
unlabelled for every Lumina head.

`data_prep/build_embedding_corpus.py` is not pre-built (166 MB). Run it when you
train embeddings. The three builders that read raw CSVs need `pandas`:
`pip install -e '.[data]'`.

## 2. Datasets

Policy: datasets built from source files contain no authored rows. Authored rows live in separate `*_synth`,
`track_signals`, `slots` folders. Report scores on real-source splits only.

| Folder | Train / val / test | Source type | Head | Use |
|---|---|---|---|---|
| `emotion_en` | 23,268 / 2,933 / 2,932 | public (GoEmotions, >=3 raters, >=50% agreement) | emotion | Train + evaluate. ANXIOUS 76 train / 9 test, CALM 72 / 9: too small to score. |
| `emotion_ar` | 6,269 / 784 / 783 | public (tweets) | emotion | 5 classes only. Tweets, not patient text. |
| `emotion_synth` | 266 / 24 / 39 | authored | emotion | OVERWHELMED, LONELY, LOW_ENERGY, ANXIOUS, CALM, DISTRESSED. Train only, weight <= 0.5. |
| `safety_addon` | 29,785 / 3,724 / 3,720 | weak supervision | safety (NORMAL/ELEVATED/HIGH) | Reddit, long NORMAL, Arabic first turns. CRISIS never assigned. |
| `safety_synth` | 63 / 8 / 8 | authored | safety | Short distress, idioms, attributions, EN/MSA/Gulf. Train only. |
| `intent` | 1 356 / 152 / 171 | authored + mixed, per-row `source_type` | intent | The whole intent corpus: authored seeds (10 meaning-families per intent) merged with the former `intent_addon` rows, split once. Score per source. |
| `track_signals` | 309 / 48 / 33 | authored | track, signal | 15 signals + NONE, 6 language varieties. Regression/coverage only. |
| `slots` | 266 / 59 / 50 | authored | primary_slot, slots | Text to state slots; test suite for a new extractor. |
| `ar_dialogue_user` | 93,600 / 11,637 / 11,776 | machine-generated | dialect | Patient turns only (MSA/EGY/GLF/LEV). Arabic corpus + dialect head. |
| `ar_patient_questions` | 13,590 / 1,697 / 1,698 | research (forum Q) | topic (14 classes) | Questions only, no answers. Topic = forum category, not diagnosis. |

Resources: `response_library.json` (13 strategies x EN/AR, 6 new-intent reply sets, track overlays: 2,380 composed
variants vs 79 sentences today), `ar_prohibited_patterns.json`, `safety_eval_seeds.jsonl` (CRISIS probes kept out of
training), `trajectories/` (160 simulated patients x 70 days, known onsets, blips, missing data).

## 3. What the three new files contained

- `mental-health.csv`: 12,565 of 20,364 posts (61.7%) already exist in `Combined_Data.csv`; removed. Labels are subreddit names.
- `data.tar`: `Label` is text in `flattened_data.csv` (ocd, anxiety, depression, other, suicide, normal) and 0-5 in `final/*.csv`. Defects:
  repeated header rows inside the data, dialect typos (EG, EEGY, E-G-Y), condition typos (Cyrillic letters in "ocd"), 692 texts shared by
  train and test, 598 by train and validation, 501 duplicate dialogues. Fixed or dropped. Provenance undocumented; data looks machine-generated.
  Assistant turns not shipped (unvetted machine text; the Arabic guard flags 323 of 120,654, 0.27%).
- `Conversation.csv`: chained small talk; used as GENERAL_CONVERSATION (capped 200) and as 300 long NORMAL windows. 90.8% of rows chain, so grouped by window.

## 4. Gaps found in the current engine (probe: 105 short authored seeds, 26 of them CRISIS; indicative, not a benchmark)

- Explicit ideation statements reaching CRISIS: EN 5/6, MSA 5/6, Gulf 4/4, Egyptian 1/3, Levantine 1/3, Arabizi 0/4.
  Seven scored NORMAL ("عايز أموت", "بدي موت", "abgha amoot", ...). Lexicon work needed, with idiom negatives ("أموت من الضحك").
- Arabic NORMAL seeds flagged ELEVATED: MSA 6/10 (4 are plain sentences with no risk wording, e.g. "سنذهب إلى الحديقة نهاية الأسبوع"), Gulf 2/4. English: 2/12, both idioms. The learned head looks out of distribution on Arabic.
- Attribution in MSA ("قال أخي إنه يريد أن يموت") reached CRISIS; English equivalent did not.
- None of the six new intents are recognised today (router returns UNKNOWN / QUESTION / GENERAL_CONVERSATION).
- `response.py` guard is English-only; Arabic replies were unchecked. `ar_guard.py` fixes that (16 self-tests pass; all library text passes both guards).

## 5. Code changes required by this pack

1. **Done.** `taxonomy.INTENTS` carries GREETING, GOODBYE, THANKS, ABOUT_BOT,
   BOT_FEEDBACK, DISENGAGE; `lumina/intent.py` has rules for all six; the intent
   head is retrained on the merged `datasets/intent` (macro-F1 0.093 → 0.457).
2. **Done.** `lumina/response.py:50` calls `ar_guard.extend(PROHIBITED_PATTERNS)`.
3. **Not done.** New composer: opener + body + question from
   `response_library.json`, anti-repeat over the last 6 replies, measured vs heard
   body choice. Nothing loads `resources/response_library.json` yet.
4. **Not done.** New slot extractor (chat text to state), tested on `slots`.
5. **Partly done.** The safety head trains on `safety` + `safety_addon` +
   `safety_synth`. Still missing: the Egyptian / Levantine / Arabizi lexicon work,
   tuned on `resources/safety_eval_seeds.jsonl`. The probe in section 4 below still
   stands — Arabizi ideation reached CRISIS 0/4.
6. **Done.** The emotion head trains on `emotion_en` + `emotion_ar` +
   `emotion_synth`, and the `weight` field is now honoured end to end: the
   synthetic rows were emitting no weight at all, so every loader defaulted them
   to 1.0 instead of 0.5. Fixed in `data_prep/build_final_synth.py`, backfilled on
   the checked-in rows, and `training/runner.py`'s `use_weights` flag — which was
   accepted and then never read — now works. Weight 0.5 beats both 0.25 and 1.0 on
   validation. Macro-F1 only moved 0.534 → 0.538: the binding constraint is the
   GoEmotions → Lumina label mapping, not the trainer. See
   `training/train_emotion.py`.

## 6. Limits

- Authored and machine-generated text is coverage, not accuracy. Scores on `*_synth`, `track_signals`, `slots` measure templates.
- Labels are subreddit, dialogue-level or authored judgements. None is clinician-rated. `clinical_validity = not_clinical` everywhere.
- Arabic dialect, Arabizi and every HIGH row were written without a native clinical reviewer. Review before use.
- The long-NORMAL texts remove the length shortcut but create a style one (Reddit comments and chat vs Reddit posts). Real diary-style NORMAL text is still missing.
- `ABOUT_BOT` reply about data visibility needs product/legal confirmation.
- Trajectory effect sizes are illustrative.
- Licences: GoEmotions Apache-2.0 (verify). All other sources unverified; confirm before shipping.
