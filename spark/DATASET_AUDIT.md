# Current planning data

The new public/synthetic planning audit is `DATASET_CARD.md` and `reports/planning-data.json`. The older FAQ audit below remains for the unchanged user CSV; its missing-label finding does not describe the new planning dataset.

---

# Dataset audit — supplied ADHD CSV

Source: `data/raw/adhd_dataset.csv`, copied byte-for-byte from the user upload. SHA-256: `7ddec51027ba48f3868501f915eff3f24bcadbc703a8abf3a4f7b69ac5d4905b`.

## Findings

| Check | Result |
|---|---|
| Format | CSV with `Question`, `Answer`, and an unnamed trailing column |
| Encoding | UTF-8 decoding fails; Windows-1252 decodes the supplied punctuation |
| Records | 10,271, excluding the header |
| English | 10,270 text-containing rows |
| Arabic / mixed | 0 / 0 |
| Empty row | 1 row missing both question and answer |
| Malformed trailing column | 5 rows with unexpected content |
| Valid normalized reference records | 10,265 |
| Quarantined records | 6, with reasons and original fields |
| Exact duplicate full Q/A pairs | 0 |
| Repeated normalized questions | 9,880 unique questions including the empty row |
| Questions with multiple answer variants | 344; these require review, not automatic contradiction labels |
| Explicit intent labels | None |
| Friction labels | None |
| Task/date/span annotations | None |
| Patient histories/outcome labels | None |
| Email/phone-like text from simple pattern checks | None found; this does not establish anonymization |
| Synthetic vs real provenance | Not supplied; cannot verify |

The data mixes ADHD information, organizational questions, trivia and unrelated academic/research questions. Some answers make medical assertions. No answer has been clinically validated by this project. **The runtime does not retrieve or repeat these FAQ answers.**

## Coverage assessment

Questions mention initiation, procrastination, overwhelm, organization, time management, distraction, routines and task completion. These are topical references, not annotations of user requests. Coverage cannot be converted into per-intent training counts or executive-function labels without review.

The CSV does not provide task spans, structured tasks, reference dates, expected dates, priority factors, energy/capacity labels, interruption state, focus outcomes or next-action ratings. It therefore cannot train or validate the requested planning pipeline as supplied. It also cannot establish Arabic, mixed-language or Arabizi support. Arabic planning in this release is an explicitly documented rule baseline, tested on development examples from the brief. Arabizi is unsupported.

Safety-related FAQ wording is not a labeled crisis dataset. The safety component instead reuses VitaMind's earlier local journal model, trained from the English/Arabic journal data supplied in this conversation. Its provenance and limitations are under `lumina/safety/`.

## Normalization and splits

`python -m training.prepare_dataset` preserves the raw file, writes UTF-8 normalized reference data, quarantines six invalid rows, and groups repeated/near-repeated questions before splitting. The character TF-IDF index is used for grouping only; it is not a runtime model or a supervised planning feature fit.

The grouping uses exact normalized questions plus an approximate nearest-neighbor graph with cosine similarity at least 0.94 and up to eight neighbors per unique question. This yielded 9,593 groups and reference partitions of 7,185 training, 1,527 validation and 1,553 test rows. Transitive connected groups remain in one split. This lexical method cannot guarantee that every semantic paraphrase is identified.

These partitions are **FAQ reference partitions**, not labeled planning train/test sets. They must not be used to claim intent accuracy, Arabic understanding, clinical performance or next-action quality. Class imbalance for the requested intent/friction labels is unmeasurable because those labels are absent.

## Training decision

Intent/friction model training and ML-versus-rule selection are blocked by missing annotations. The delivered assistant uses rule baselines for those components and exposes that fact in its structured output. Metadata files explicitly say `NOT_TRAINED_MISSING_LABELS`; there are no fabricated intent-model weights.

The training script accepts a future reviewed JSONL dataset, rejects the FAQ CSV, keeps supplied meaning groups separated, fits vocabulary on training only, compares logistic regression / linear SVM / Naive Bayes against rules on validation, and evaluates the selected approach on test only afterward. It retains rules if no model improves validation macro F1.

See `NEEDED_DATA.md` for the exact missing data and annotation format. More duplicated FAQ rows would not fill this gap.
