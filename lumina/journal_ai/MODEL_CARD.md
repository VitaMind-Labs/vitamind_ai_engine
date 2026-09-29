# Journal Sentinel model card

## Identity and scope

A from-scratch multi-task linear classifier with shared TF-IDF word/character features. The delivered artifact is actually trained and is connected to `JournalSentinel`. There are no pretrained embeddings, transformer weights, fine-tuning steps or API keys.

- Artifact version: `journal-linear-seed42-v1`.
- Learned input vocabulary: 7,404 features.
- Six prediction tasks; the category task uses nine binary heads.
- 28 SGD epochs, seed 42; weights initialized to zero.
- High-tier loss multiplier: 3. Validation temperature search: 0.5–4.0.
- Weights SHA-256: `f403115b36f2ec5319b8884e39d098ea6aa2064478a05703a7497f89042cd9bc`.
- Built and tested with Python 3.12.14 and NumPy 2.3.5. Runtime supports Python 3.10+ with NumPy 1.26–2.x.

## Data audit

The supplied files contain 4,000 training rows, 500 validation rows and 48 challenge examples. The claimed separate 500-row test file and XLM-R checkpoint were not uploaded. Source files are preserved unchanged under `data/source/`.

The original train/validation files share no exact core IDs, but share **5 base meaning families**: h2, h4, si2, si6, si7. For example, a self variant and a past/other variant of one meaning could land on opposite sides. The preparation script groups these variants together.

Combining the original train/validation rows and normalizing duplicates removes 486 duplicate rows. 0 conflicting texts were quarantined. No diagnosis labels were guessed, translated or relabeled.

| Partition | Rows | Meaning families | English | Arabic | Mixed |
|---|---:|---:|---:|---:|---:|
| train | 2724 | 37 | 1171 | 1015 | 538 |
| val | 661 | 11 | 285 | 240 | 136 |
| test | 629 | 11 | 285 | 223 | 121 |

Vocabulary/IDF and learned weights use training rows only. Temperature selection uses validation rows. Base meaning families and normalized texts do not overlap between the generated partitions. The challenge set does not fit features, weights or temperatures.

## Measured results

The learned classifier and the full rule-assisted pipeline are reported separately. They must not be described as the same result.

| Evaluation | Rows | Model-only tier accuracy | Full pipeline tier accuracy | Pipeline high recall | Pipeline false-high rate |
|---|---:|---:|---:|---:|---:|
| Grouped validation | 661 | 43.6% | 56.0% | 98.7% | 0.5% |
| Grouped test | 629 | 52.8% | 73.4% | 97.1% | 0.0% |
| Uploaded regression set | 48 | 58.3% | 83.3% | 100.0% | 0.0% |

False-high rate means the fraction of examples labeled below high that the pipeline classifies high. The source labels are synthetic policy labels, not verified clinical outcomes.

| Grouped test language | Rows | Model-only accuracy | Pipeline accuracy |
|---|---:|---:|---:|
| English | 285 | 43.5% | 64.6% |
| Arabic | 223 | 65.0% | 83.4% |
| Mixed | 121 | 52.1% | 76.0% |

The uploaded high-tier regression gate currently passes **6/6** cases. Rules were refined after inspecting that benchmark, so this is a regression check, not a blinded estimate of generalization. The complete evaluation includes all disagreements and confusion matrices in `reports/evaluation.json`.

## Limits that matter

- Model-only performance on unseen meaning families is modest. Stronger crisis-recall numbers largely come from the explicit safety policy; they do not demonstrate deep language understanding.
- Thousands of augmented rows represent a small number of authored meanings. More copies of these templates would not establish clinical reliability.
- All datasets are synthetic and share authorship/design assumptions. There is no real-patient validation or bilingual clinical review.
- The labels conflate some passive wishes with high intent and label some other-person concerns low. The pipeline intentionally provides support without assigning that concern to the writer.
- Finite rules can miss paraphrases, sarcasm, attribution and long-distance context. Arabic dialect coverage is limited to what appears in the supplied corpus and authored rules.
- The classifier score is not a calibrated probability of harm. Validation temperature fitting on this small synthetic set does not make it clinical risk calibration.
- Personal signatures are normalized phrase matches, not semantic relapse detection. Counts require correctly supplied saved-entry history.
- Responses are fixed development templates, not generated clinical advice. No diagnosis or treatment recommendation is produced.

**Use this version for local development and further evaluation. It is not ready to act as the sole crisis detector for real patients.**

## Retraining and reproducibility

From the `lumina/` folder, run `python -m journal_ai.training.train`, then `python -m journal_ai.training.evaluate --strict` and `python -m pytest journal_ai/tests -q`. Every training run starts from zero and overwrites the model artifact. Preserve your current folder first if you want to compare versions. Recreate `JournalSentinel` or restart the process after training.

Prepared partitions are regenerated deterministically from source files and the seed. Model results may vary slightly by NumPy/platform version. Source hashes and partition membership are stored in `reports/data-audit.json`.

Future improvement requires independently authored, more diverse labeled meanings and expert-reviewed context labels, then a new untouched evaluation set. Changes to the rules should be evaluated separately from changes to learned weights.
