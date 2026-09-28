# Use of the uploaded Arabic dataset

Source: [Arabic Mental Health Dataset, Zenodo record 20568736](https://zenodo.org/records/20568736),
attributed by its publisher to Ahmed Taha Shiha. The three CSV MD5 checksums in
your uploaded ZIP exactly match the published files. The source ZIP is not modified.

| Original split | Rows |
| --- | ---: |
| train.csv | 21,780 |
| validation.csv | 3,844 |
| test.csv | 4,522 |

All three files use a numeric `Label` column with values 0–5. They do not include
a label key. The source description discusses anxiety, OCD, depression,
suicidality, normal and other cases. It does not provide verified target labels
for Mira's ADHD, bipolar and psychosis screening pathways. Clinical labeling
claims on that page have not been independently validated here.

Consequently, source numeric labels are **not used as supervised diagnosis
targets**. The imported text trains vocabulary/IDF only. Mira's supervised
English/Arabic routing examples are separately identified as synthetic.
Correct, independently reviewed labels for the desired targets are still needed
before this source could support supervised learning of those targets.

## Actual preparation

- Read `text`, preserving negation. Do not use `cleaned_text`, which removes some
  words and combines speakers.
- Take only the text before the first `|`. This assumes the source is user-first;
  the CSV has no explicit speaker annotations, so this assumption needs review.
  Do not use or replay subsequent chatbot answers, including unsafe advice.
- Exclude chatbot-related questions, malformed or invalid header/flag/dialect
  rows, empty/very short/very long non-Arabic entries, and simple contact-identifier
  patterns. This is not certified anonymization.
- Deduplicate normalized openings. Remove training openings whose normalized text
  occurs in source validation/test. This is exact matching, not a guarantee
  against translated/near-duplicate patients or scenarios across the original files.
- Retain **17,406** training openings. Do not fit on source validation/test.

The learner fits word/character vocabularies and IDF on labeled training text
plus these unlabeled Arabic openings. Classification coefficients fit only the
labeled synthetic English/Arabic training records. Vocabulary adaptation is
not equivalent to learning a diagnostic label or generating conversation.

The imported corpus and full filtering audit are in
`data/arabic_adaptation.jsonl` and `data/arabic_adaptation.audit.json` (project-root
paths). Source hashes and per-reason exclusion counts make the import reproducible.

The source contains MSA, Gulf, Egyptian and Levantine dialect tags. Vocabulary
exposure does not establish reliable dialect understanding. The supervised
Arabic fixtures and report language use primarily Modern Standard Arabic.
No French or Tunisian training set is added.
